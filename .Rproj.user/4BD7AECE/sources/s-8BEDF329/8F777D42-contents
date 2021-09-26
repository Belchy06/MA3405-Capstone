

importedData <- read.csv(file = "csv_result-Autism-Adult-Data.csv")

cleanData <- importedData[apply(importedData, 1, function(row) all(row != '?')), ]
augmentedData <- cleanData[ , !(names(cleanData) %in% c('id', 'age_desc'))]
