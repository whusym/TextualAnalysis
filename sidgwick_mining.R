library(rvest)
library(stringr)

Sidgwick_work <- list("http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.1;toc.depth=2;toc.id=div.britphil.v59.1;hit.rank=0;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.13;toc.id=div.britphil.v59.13;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.23;toc.id=div.britphil.v59.23;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.30;toc.id=div.britphil.v59.30;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.45;toc.id=div.britphil.v59.45;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.61.xml;chunk.id=div.britphil.v59.52;toc.id=div.britphil.v59.52;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.62.xml;chunk.id=div.britphil.v60.1;toc.depth=2;toc.id=div.britphil.v60.1;hit.rank=0;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.62.xml;chunk.id=div.britphil.v60.9;toc.id=div.britphil.v60.9;brand=default",
                  "http://pm.nlx.com/xtf/view?docId=britphil/britphil.62.xml;chunk.id=div.britphil.v60.14;toc.id=div.britphil.v60.14;brand=default")


scraping <- function(name, output, list){
  for (i in list){
    sink(file = output, append = TRUE)  #write result in the file
    # loop through the file list
    name <- read_html(i)
    
    name %>%   # scrape the title of the work
      html_nodes(".volume_title") %>%
      html_text() %>%
      cat() # or writeLines()
    cat("\n")   #separate the title and the text
    
    name %>%
      html_nodes("p") %>%
      html_text() %>%
      cat() %>% #writeLines() %>%
      str_replace_all("\\\n","")  #remove the new line at the end of each paragraph
    cat("\n")   #separate each article
    sink()   #clean sink
  }
}

scraping(Sidgwick, "sidgwick-new.txt", Sidgwick_work)