```
remove redundant features (highly correlated of less granularity)

for each column{
   if dtype == object{
       find and store indices of missing values (create boolean)
       binary encode column
   }
   else if dtype == float or int{
       find and store indices of missing values (create boolean)
   }
}

for each column{

    copy dataframe and drop rows with missing values in column
    drop column from new dataframe
    perform knn on new dataframe

    if dtype == object{
        predict cluster for dropped rows
        find distribution of missing value in all clusters
        find value with max probability for each cluster
        for each dropped row{
            set missing value in dropped column to max probability value from cluster 
        }
    }
    else if dtype == float{
        predict cluster for dropped rows
        find mean value with for each cluster
        for each dropped row{
            set missing value in dropped column to mean value from cluster 
        }
    }
}
```