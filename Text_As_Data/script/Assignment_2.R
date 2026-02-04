# Set Up
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)


circle_raw <- read_file("data/text/A07594__Circle_of_Commerce.txt")
free_raw   <- read_file("data/text/B14801__Free_Trade.txt")

texts <- tibble(
  doc_title = c("Circle of Commerce", "Free Trade"),
  text = c(circle_raw, free_raw)
)

# Normalization
texts <- texts %>%
  mutate(
    text_clean = text %>%
      str_replace_all("ſ", "s") %>%     # long s normalization
      str_replace_all("\\s+", " ") %>%  # collapse whitespace
      str_to_lower()
  )

# Raw Count
## Tokenize & remove stopwords

data("stop_words")

tokens <- texts %>%
  unnest_tokens(word, text_clean) %>%
  anti_join(stop_words, by = "word")

## joint Bing
bing <- get_sentiments("bing")

sentiment_words <- tokens %>%
  inner_join(bing, by = "word")

## Raw sentiment totals per document
raw_sentiment_summary <- sentiment_words %>%
  count(doc_title, sentiment) %>%
  ## values_fill = 0 ensures that missing sentiment categories are treated as 0
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  
mutate(
    net_sentiment_raw = positive - negative
  )

raw_sentiment_summary

# TF-IDF
## Compute TF-IDF
word_counts <- tokens %>%
  count(doc_title, word)

tfidf_tbl <- word_counts %>%
  bind_tf_idf(term = word, document = doc_title, n = n)

sentiment_words_tfidf <- tfidf_tbl %>%
  inner_join(bing, by = "word")

## Totals
tfidf_sentiment_summary <- sentiment_words_tfidf %>%
  group_by(doc_title) %>%
  summarise(
    tfidf_positive = sum(tf_idf[sentiment == "positive"], na.rm = TRUE),
    tfidf_negative = sum(tf_idf[sentiment == "negative"], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    tfidf_positive = replace_na(tfidf_positive, 0),
    tfidf_negative = replace_na(tfidf_negative, 0),
    net_sentiment_tfidf = tfidf_positive - tfidf_negative
  )

tfidf_sentiment_summary

### Comparison table
final_sentiment_comparison <- raw_sentiment_summary %>%
  left_join(tfidf_sentiment_summary, by = "doc_title")

final_sentiment_comparison

write_csv(final_sentiment_comparison,
          "sentiment_raw_vs_tfidf.csv")



## TF-IDF sentiment words for Circle of Commerce
sentiment_words_tfidf %>%
  filter(doc_title == "Circle of Commerce",
         sentiment == "negative") %>%
  arrange(desc(tf_idf)) %>%
  select(word, tf_idf) %>%
  slice_head(n = 10)

## TF-IDF sentiment words for Free Trade
sentiment_words_tfidf %>%
  filter(doc_title == "Free Trade",
         sentiment == "negative") %>%
  arrange(desc(tf_idf)) %>%
  select(word, tf_idf) %>%
  slice_head(n = 10)

##################
##################

total_tfidf_sentiment <- sentiment_words_tfidf %>%
  group_by(doc_title) %>%
  summarise(
    total_tfidf_sentiment = sum(abs(tf_idf)),
    .groups = "drop"
  )

top5_sentiment_words <- sentiment_words_tfidf %>%
  mutate(tfidf_abs = abs(tf_idf)) %>%
  group_by(doc_title) %>%
  slice_max(tfidf_abs, n = 5) %>%
  summarise(
    top5_tfidf = sum(tfidf_abs),
    .groups = "drop"
  )

sentiment_concentration <- total_tfidf_sentiment %>%
  left_join(top5_sentiment_words, by = "doc_title") %>%
  mutate(
    proportion_top5 = top5_tfidf / total_tfidf_sentiment
  )

sentiment_concentration
