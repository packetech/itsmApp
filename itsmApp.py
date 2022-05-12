import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
set(stopwords.words('english'))
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    raw_text = request.form['text']
    lemmatizer = WordNetLemmatizer()
	stemmer = PorterStemmer() 

    def preprocess(sentence):
    	sentence=str(sentence)
    	sentence = sentence.lower()
    	sentence=sentence.replace('{html}',"") 
    	cleanr = re.compile('<.*?>')
    	cleantext = re.sub(cleanr, '', sentence)
    	rem_url=re.sub(r'http\S+', '',cleantext)
    	rem_num = re.sub('[0-9]+', '', rem_url)
    	tokenizer = RegexpTokenizer(r'\w+')
    	tokens = tokenizer.tokenize(rem_num)  
    	filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    	stem_words=[stemmer.stem(w) for w in filtered_words]
    	lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    	#return " ".join(filtered_words)
    	return " ".join(lemma_words)
    
    raw_text = (lambda s:preprocess(s))(raw_text)

    #Apply Tokenizer and get indices for words
	feature = 5000
	maxlen = 25 # Max Lenth of sentence to pad to
	trunc_type = 'post'
	padding_type = 'post'
	oov_tok = '<OOV>'

	tokenizer = Tokenizer(num_words=feature, oov_token=oov_tok)
	tokenizer.fit_on_texts(list(raw_text))
	X_raw = tokenizer.texts_to_sequences(raw_text)
	X_raw = pad_sequences(X_raw, maxlen = maxlen, value=0, padding = padding_type, truncating = trunc_type)
	itsm_class_prob = loaded_model.predict(X_raw, batch_size=64, verbose=2)[0]
	itsm_class_pred = np.rint(itsm_class_prob)

	output1 = pd.DataFrame(itsm_class_pred, columns=['class_predicted'])
	output2 = pd.DataFrame(np.round(itsm_class_prob, 2), columns=['probability'])
	output3 = output1. join(output2)


	itsm_class_pred = pd.DataFrame(itsm_class_pred, columns=['pred'])
	row = itsm_class_pred.loc[itsm_class_pred['pred'] == 1]

	if len(row) == 0:
    	#print('Manual classification necessary !')
    	outputPred = 'Manual classification necessary !'
    
	elif len(row) > 1: 
    	maxVal = output3['probability'].max()
    	ind2 = output3.class_predicted[output3['probability'] == maxVal]
    	grppy_df = pd.read_pickle("./grppy_df.pkl")
    	#print("The logged ticket is classified as: {}".format(grppy_df[ind2.index[0]][0])); print("And its probability is : {}".format(maxVal))
		outputPred = grppy_df[ind2.index[0]][0]

	else:
    	ind = row.index[0]
    	maxVal = output3['probability'].max()
    	grppy_df = pd.read_pickle("./grppy_df.pkl") 
    	#print("The logged ticket is classified as: {}".format(grppy_df[ind][0])) ; print("And its probability is : {}".format(round(maxVal, 2)))
    	outputPred = grppy_df[ind][0]



    return render_template('index.html', prediction_text='The logged ticket is classified as {}'.format(outputPred))


if __name__ == "__main__":
    app.run(debug=True)

