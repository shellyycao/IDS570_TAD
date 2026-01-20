# Load Package:
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)

# Load File
file_a <- "data/text/A07594__Circle_of_Commerce.txt"
file_b <- "data/text/B14801__Free_Trade.txt"

# Read the raw text files into R
text_a <- read_file(file_a)
text_b <- read_file(file_b)

# Combine into a tibble for tidytext workflows
texts <- tibble(
  doc_title = c("Text A", "Text B"),
  text = c(text_a, text_b)
)

texts

# Create a diagnostics table 

corpus_diag <- texts %>%
mutate(n_chars = str_length(text)) %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_to_lower(word)) %>%
  group_by(doc_title) %>%
  summarise(
    n_chars = first(n_chars),
    n_word_tokens = n(),
    n_word_types = n_distinct(word),
    .groups = "drop"
  )

corpus_diag

# Stop words:
data("stop_words")

custom_stopwords <- tibble(
  word = c("vnto", "haue", "doo", "hath", "bee", "ye", "thee")
)

all_stopwords <- bind_rows(stop_words, custom_stopwords) %>%
  distinct(word)

# Tokenize:
word_counts <- texts %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_to_lower(word)) %>%
  anti_join(all_stopwords, by = "word") %>%
  count(doc_title, word, sort = TRUE)

word_counts

# Compute document length:
doc_lengths <- word_counts %>%
  group_by(doc_title) %>%
  summarise(total_words = sum(n), .groups = "drop")

doc_lengths

# Create normalized frequencies:
word_counts_normalized <- word_counts %>%
  left_join(doc_lengths, by = "doc_title") %>%
  mutate(relative_freq = n / total_words)

word_counts_normalized

# Trade:
word_counts_normalized %>%
  filter(word == "trade")

# Plot:
plot_n_words <- 20

word_comp_tbl <- word_counts %>%
  pivot_wider(
    names_from = doc_title,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(max_n = pmax(`Text A`, `Text B`)) %>%
  arrange(desc(max_n))

top_words <- word_comp_tbl %>%
  slice_head(n = plot_n_words) %>%
  select(word)

word_plot_data <- word_counts_normalized %>%
  semi_join(top_words, by = "word") %>%
  mutate(word = fct_reorder(word, relative_freq, .fun = max))

ggplot(word_plot_data, aes(x = relative_freq, y = word)) +
  geom_col() +
  facet_wrap(~ doc_title, scales = "free_x") +
  labs(
    title = "Most frequent words (normalized by document length)",
    subtitle = paste0(
      "Top ", plot_n_words,
      " words selected using raw counts (Week 02), plotted as relative frequency"
    ),
    x = "Relative frequency of word",
    y = NULL
  ) +
  theme_minimal()

