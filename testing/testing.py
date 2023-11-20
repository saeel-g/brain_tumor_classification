import os
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, roc_curve, confusion_matrix, accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import glob
from tensorflow.keras.models import load_model
import csv

tf.keras.backend.clear_session()

SIZE=256

test = pd.read_csv('../testing/test_dataset/test_labels.csv')

df_test = pd.DataFrame(test)

test_labels=df_test['IMG'].values

test_image = []
for i in tqdm(test_labels):
    img = image.load_img(f'../testing/test_dataset/{str(i)}', target_size=(SIZE, SIZE,3))
    img = image.img_to_array(img)
    img = img[:, :, 0]  # Take only one channel (assuming it's a grayscale image)
    img = img / 255
    img = np.expand_dims(img, axis=-1)  # Add a singleton dimension for the channel
    test_image.append(img)

path='../trained models/'
csv_file_path = '../testing/results.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Model', 'Total_TP', 'Total_TN', 'Total_FP', 'Total_FN', 'Total_PR', 'Total_RC', 'Total_F1', 'Total_ACC', 'AUC'])

    for filename in os.listdir(path):
        file_path=os.path.join(path,filename)
        model = tf.keras.models.load_model(f'{file_path}')
        print('********Results of: ',filename,'************')


        X_test = np.array(test_image)
        y_test = np.array(test.drop(['IMG'],axis=1))
        print(X_test.shape)
        print(y_test.shape)



        batch_size = 8  # Set your desired batch size
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        _, acc = model.evaluate(test_dataset)

        y_pred1 = model.predict(X_test)

        roc=roc_auc_score(y_test, y_pred1, multi_class='ovr', average='micro')

        print("AUC= ",roc)


        df_test=y_test
        y_test_df=df_test.argmax(axis=1)

        df_pred=y_pred1
        y_pred_df=df_pred.argmax(axis=1)


        conf_matrix = confusion_matrix(y_test_df, y_pred_df)
        print("Confusion Matrix:\n", conf_matrix)

        # Choose a specific class as the positive class (adjust class_index accordingly)
        true_positive=[]
        true_negative=[]
        false_positive=[]
        false_negative=[]
        for i in range(4):
            positive_class_index = i

            tn, fp, fn, tp = confusion_matrix(y_test_df == positive_class_index, y_pred_df == positive_class_index).ravel()
            print("True Positive:", tp, "False Positive:", fp, "True Negative:", tn, "False Negative:", fn)
            print('*****'*5)

            true_positive.append(tp)
            true_negative.append(tn)
            false_positive.append(fp)
            false_negative.append(fn)

        print(true_positive)
        print(true_negative)
        print(false_positive)
        print(false_negative)

        total_tp=np.sum(true_positive)
        total_tn=np.sum(true_negative)
        total_fp=np.sum(false_positive)
        total_fn=np.sum(false_negative)

        print(total_tp)
        print(total_tn)
        print(total_fp)
        print(total_fn)


        # Calculate Precision, Recall, F1-Score, and Accuracy for the entire model
        total_PR = total_tp / (total_tp + total_fp)
        total_RC = total_tp / (total_tp + total_fn)
        total_F1 = 2 * (total_PR * total_RC) / (total_PR + total_RC)
        total_ACC = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)

        print("\nMetrics for the Entire Model:")
        print("Total True Positive:", total_tp)
        print("Total True Negative:", total_tn)
        print("Total False Positive:", total_fp)
        print("Total False Negative:", total_fn)
        print("\nPrecision: ", total_PR, "\nRecall: ", total_RC, "\nF1-Score", total_F1, "\nAccuracy:", total_ACC, "\nAUC", roc)
        print('*'*50)
        writer.writerow([filename, total_tp, total_tn, total_fp, total_fn, total_PR, total_RC, total_F1, total_ACC, roc])
print(f"Results saved to {csv_file_path}")