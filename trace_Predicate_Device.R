# ===================================================================
# FDA 510(k) 数字疗法产品 Predicate Device 网络分析
# 改进版（去除构造PDF链接 + 即时保存CSV + 新字段结构）
# ===================================================================

rm(list = ls())
graphics.off()

# ----------------------------
# 依赖包安装与加载
# ----------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse, readxl, httr, xml2, rvest, tesseract, pdftools, jsonlite,
  magick, igraph, ggraph, stringi, progress, future, furrr, fs
)

workers <- max(1, parallel::detectCores() - 1)
plan(multisession, workers = workers)

# ----------------------------
# 全局参数
# ----------------------------
options(timeout = 30)
user_agent_str <- "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
eng <- tesseract("eng", options = list(tessedit_pageseg_mode = 6))

log_dir <- "DTx_Analysis_Logs"
dir_create(log_dir, showWarnings = FALSE)
log_file <- file.path(log_dir, paste0("dtx_analysis_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
sink(log_file, split = TRUE)
cat("日志文件:", log_file, "\n")

# ----------------------------
# 数据输入
# ---------------------------
excel_path <- "E:/一匹午马/DTx/小张的方案呢？/研究/TracetotheSource/251029-涂涂改改完善/数字疗法产品reTrace_4-5.xlsx"
if (!file.exists(excel_path)) stop("找不到 Excel 文件。")
df <- read_excel(excel_path, sheet = 1)
cat("Excel 读取成功:", nrow(df), "行\n")

# ----------------------------
# HTTP 工具函数
# ----------------------------
safe_get <- function(url, query = NULL, max_attempts = 4, timeout_sec = 60) {
  tryCatch({
    res <- RETRY("GET", url, query = query, times = max_attempts,
                 pause_base = 2, pause_cap = 60,
                 terminate_on = c(200), timeout(timeout_sec),
                 user_agent(user_agent_str), quiet = TRUE)
    if (inherits(res, "response") && !http_error(res)) return(res)
    NULL
  }, error = function(e) { cat("safe_get 错误:", e$message, "\n"); NULL })
}

# ----------------------------
# PDF 文本提取（含 OCR）
# ----------------------------
extract_text_from_pdf <- function(url) {
  cat("处理 PDF:", url, "\n")
  tryCatch({
    tmp_pdf <- tempfile(fileext = ".pdf")
    download.file(url, tmp_pdf, mode = "wb", quiet = TRUE)
    if (!file.exists(tmp_pdf) || file.size(tmp_pdf) < 1024) return("")
    
    text <- tryCatch(pdf_text(tmp_pdf) %>% paste(collapse = "\n"), error = function(e) "")
    if (nchar(text) < 1000) {
      info <- tryCatch(pdf_info(tmp_pdf), error = function(e) NULL)
      if (!is.null(info)) {
        ocr_text <- character(0)
        for (pg in seq_len(info$pages)) {
          img <- tryCatch({
            img_file <- tempfile(fileext = ".png")
            pdf_convert(tmp_pdf, format = "png", pages = pg, dpi = 300, filenames = img_file)
            image_read(img_file) %>%
              image_convert(type = "Grayscale") %>%
              image_normalize() %>%
              image_trim()
          }, error = function(e) NULL)
          if (!is.null(img)) {
            ocr_text <- c(ocr_text, ocr(img, engine = eng))
          }
        }
        text <- paste(text, paste(ocr_text, collapse = "\n"), sep = "\n")
      }
    }
    unlink(tmp_pdf)
    text
  }, error = function(e) { cat("extract_text_from_pdf 出错:", e$message, "\n"); "" })
}

# ----------------------------
# predicate 提取
# ----------------------------
extract_predicate_numbers <- function(text) {
  if (is.null(text) || nchar(text) < 100) return(character(0))
  str_extract_all(text, "(K|DEN|BK|P)\\d{6}") %>% unlist() %>% unique()
}

try_alternative_predicate_extraction <- function(k_number, text = "") {
  cat("尝试替代 predicate 提取:", k_number, "\n")
  api_url <- "https://api.fda.gov/device/510k.json"
  res <- safe_get(api_url, query = list(search = paste0("k_number:", k_number), limit = 1))
  if (!is.null(res)) {
    parsed <- tryCatch(content(res, "parsed"), error = function(e) NULL)
    if (!is.null(parsed) && length(parsed$results) > 0) {
      preds <- unlist(parsed$results[[1]]$predicate_device)
      preds <- na.omit(str_extract(preds, "(K|DEN|BK|P)\\d{6}"))
      if (length(preds) > 0) {
        cat("API 找到 predicate:", paste(preds, collapse = ","), "\n")
        return(unique(preds))
      }
    }
  }
  character(0)
}

# ----------------------------
# 获取 FDA 文档链接
# ----------------------------
get_fda_doc_links <- function(k_number) {
  cat("获取文档链接:", k_number, "\n")
  api_url <- "https://api.fda.gov/device/510k.json"
  res <- safe_get(api_url, query = list(search = paste0("k_number:", k_number), limit = 1))
  if (!is.null(res)) {
    parsed <- tryCatch(content(res, "parsed"), error = function(e) NULL)
    if (!is.null(parsed) && length(parsed$results) > 0) {
      device <- parsed$results[[1]]
      if (!is.null(device$documents)) {
        urls <- sapply(device$documents, function(x) x$url)
        urls <- urls[grepl("\\.pdf$", urls, ignore.case = TRUE)]
        if (length(urls) > 0) {
          cat("通过 API 找到文档:", paste(urls, collapse = ", "), "\n")
          return(unique(urls))
        }
      }
    }
  }
  
  # 策略2：网页爬取
  page_url <- "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
  res2 <- safe_get(page_url, query = list(ID = k_number))
  if (!is.null(res2)) {
    html_text <- tryCatch(content(res2, "text", encoding = "UTF-8"), error = function(e) NULL)
    if (!is.null(html_text)) {
      links <- read_html(html_text) %>%
        html_nodes("a") %>%
        html_attr("href") %>%
        na.omit() %>% unique()
      pdfs <- links[grepl("\\.pdf$", links, ignore.case = TRUE)]
      if (length(pdfs) > 0) {
        pdfs <- ifelse(grepl("^http", pdfs), pdfs, paste0("https://www.accessdata.fda.gov", pdfs))
        cat("网页爬取到 PDF:", paste(pdfs, collapse = ", "), "\n")
        return(unique(pdfs))
      }
    }
  }
  cat("未找到文档链接。\n")
  NULL
}

# ----------------------------
# 结果构造 + 即时写入
# ----------------------------
create_result_tibble <- function(device, depth, predicate_of, software_intro) {
  tibble(
    device = device,
    depth = depth,
    predicate_of = predicate_of,
    software_intro = software_intro
  )
}

# ----------------------------
# 递归查找并即时保存
# ----------------------------
find_predicate_network <- function(k_number, depth = 0, predicate_of = NA,
                                   max_depth = 20, visited = character(0),
                                   out_path = NULL) {
  if (depth > max_depth) return()
  if (k_number %in% visited) return()
  visited <- c(visited, k_number)
  
  cat("\n=== 处理设备:", k_number, " | 深度:", depth, " ===\n")
  
  doc_links <- get_fda_doc_links(k_number)
  if (is.null(doc_links) || length(doc_links) == 0) {
    res <- create_result_tibble(k_number, depth, predicate_of, FALSE)
    if (!is.null(out_path)) write_csv(res, out_path, append = TRUE)
    return()
  }
  
  all_text <- ""
  software_intro <- FALSE
  predicates <- character(0)
  
  for (link in doc_links) {
    doc_text <- extract_text_from_pdf(link)
    all_text <- paste(all_text, doc_text, sep = "\n")
    if (str_detect(doc_text, regex("software|algorithm|AI|DTx|digital therap", ignore_case = TRUE))) {
      software_intro <- TRUE
    }
    preds <- extract_predicate_numbers(doc_text)
    predicates <- unique(c(predicates, preds))
  }
  
  if (length(predicates) == 0)
    predicates <- try_alternative_predicate_extraction(k_number, all_text)
  
  # 当前节点写入
  res <- create_result_tibble(k_number, depth, predicate_of, software_intro)
  if (!is.null(out_path)) write_csv(res, out_path, append = TRUE)
  
  # 递归下一层
  for (pred in predicates) {
    find_predicate_network(pred, depth + 1, k_number,
                           max_depth, visited, out_path)
  }
}

# ----------------------------
# 主函数
# ----------------------------
main_analysis <- function() {
  valid_k <- df$注册证编号[grepl("^K\\d{6}$", df$注册证编号)]
  valid_k <- unique(valid_k)
  if (length(valid_k) == 0) {
    cat("未找到有效 K 号。\n")
    return(NULL)
  }
  
  results_dir <- "DTx_Predicate_Analysis_Results"
  dir_create(results_dir, showWarnings = FALSE)
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_file <- file.path(results_dir, paste0("predicate_results_", timestamp, ".csv"))
  
  # 写入标题行
  write_csv(tibble(device = character(), depth = integer(),
                   predicate_of = character(), software_intro = logical()),
            out_file, append = FALSE)
  
  for (k in valid_k) {
    find_predicate_network(k, depth = 0, predicate_of = NA, out_path = out_file)
  }
  
  cat("结果文件生成：", out_file, "\n")
  return(out_file)
}

# ----------------------------
# 运行
# ----------------------------
final_csv <- main_analysis()
sink()
cat("脚本执行结束。日志存放于:", log_file, "\n")
