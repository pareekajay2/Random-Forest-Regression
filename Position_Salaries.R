dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

install.packages('randomForest')
library(randomForest)

set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)

y_pred = predict(regressor, data.frame(Level = 6.5))

library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'Red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            color = 'Blue') +
  ggtitle('Salary vs Level') +
  xlab('Level') +
  ylab('Salary')