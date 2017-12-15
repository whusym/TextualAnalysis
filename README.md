## Textual Analysis of Modern British Philosophers
---------------------------------------------------

- In this project, I analyze the works of modern British empiricists, including Thomas Hobbes, John Locke, George Berkeley, David Hume, Jeremy Bentham, John Stuart Mill and Henry Sidgwick.
- How to use: run analysis.py. You can also open analysis.ipynb to see the details and results in Jupyter Notebook
- Dependencies: nltk, sklearn, numpy, pandas, matplotlib, textblob (for sentiment analysis), gensim (for LDA modeling), Jupyter notebook (if users want to check the notebook)
- Add text files into /authorCorpus folder and change AUTHOR_FILES variable to get keyword frequencies for each file.
  ** Plotting results of keyword frequencies are in the /keywordResults folder
- Add text files into /authorBooks folder and change BOOK_LIST to get classification results.
- Files in the TEST_FILES list in the program are for testing the classification results. These files are also in the /authorBooks folder
- The R file is for text scraping. I only provide an example for Sidgwick. Others' files are available too. Just change the URLs in the list.
