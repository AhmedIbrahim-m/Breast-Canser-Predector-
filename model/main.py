import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle as pickle


from sklearn.metrics import accuracy_score, confusion_matrix


def  get_clean():
    data = pd.read_csv("C:/Users/Administrator/Downloads/archive (5)/data.csv")
  
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1 , 'B': 0})
    return data 
   
   
   
def model_(data):
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis'] 
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=42 )
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    #test 
    y_pred = model.predict(x_test )
    print("Acc" , accuracy_score(y_test , y_pred))
    print("classification rep " , classification_report(y_test, y_pred))
    
    return model , scaler



def main():
    data = get_clean()
    model ,scaler = model_(data)
    
    with open('canser_pred_logistic/model.pkl', 'wb') as f:
         pickle.dump(model, f)

    with open('canser_pred_logistic/scaler.pkl', 'wb') as f:
         pickle.dump(scaler, f) 
    

if __name__ == '__main__':
    
    main() 