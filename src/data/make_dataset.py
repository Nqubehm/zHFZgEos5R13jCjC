# Imports
from sklearn.preprocessing import LabelEncoder

# Check null counts in all our features
data.isnull().sum()

# Confirm the new data size
data.shape

# Remove duplicates if any
data.drop_duplicates(inplace=True)

#### Convert Categorical To Numerical Data

# Get only categorical features
cat_data = data.select_dtypes(exclude='int64')

# Exclude the dependent variable
cat_data = cat_data.drop('y', axis=1)

# Instantiate label encoder
encoder = LabelEncoder()

# encode all categorical to numericals
for i in cat_data.columns:
    data[i] = encoder.fit_transform(data[i])
    
# Check our new data 
data.head()   
