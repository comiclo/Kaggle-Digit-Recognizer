import pandas as pd
from preprocessing import load_data
from keras.models import load_model

model = load_model('./mnist.model')

train_data, train_labels, test_data = load_data()

predictions = model.predict_classes(test_data)
df = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'label': predictions})
df.to_csv('submission.csv', index=False)
