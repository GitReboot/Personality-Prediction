from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import re
import string
import nltk
from unidecode import unidecode
import csv
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import fitz
from tkinter import *
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk

# Downloading stopwords and punkt from Natural language toolkit
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Functions for pre-processing of the data that is remove urls, punctuations, numbers etc.
def replace_sep(text):
    text = text.replace("|||", ' ')
    return text


def remove_url(text):
    text = re.sub(r'https?:*?[\s+]', '', text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_numbers(text):
    text = re.sub(r'[0-9]', '', text)
    return text


def convert_lower(text):
    text = text.lower()
    return text


def extra(text):
    text = text.replace("  ", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text


# Using nltk stop words to remove common words not required in processing like a, an, the & etc.
Stopwords = set(stopwords.words("english"))


def stop_words(text):
    tweet_tokens = word_tokenize(text)
    filtered_words = [w for w in tweet_tokens if not w in Stopwords]
    return " ".join(filtered_words)


# Applying lemmatization
def lemmatization(text):
    tokenized_text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(a) for a in tokenized_text])
    return text


# Doing the pre-processing of data by the functions defined above
def pre_process(text):
    text = replace_sep(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = convert_lower(text)
    text = extra(text)
    text = stop_words(text)
    text = lemmatization(text)
    return text


# Tokenizing the data we retrieve from LinkedIn.
# Defining the various emojis and emoticons and creating their regex patterns.
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


# Pre-processing the tokenized data
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# Using unidecode to remove all the non ascii characters from our string.
def preproc(s):
    s = unidecode(s)
    POSTagger = preprocess(s)
    # print(POSTagger)

    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokenize(tweet)
    filtered_sentence = []
    for w in POSTagger:
        if w not in stop_words:
            filtered_sentence.append(w)
    # print(word_tokens)
    # print(filtered_sentence)
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    # print(stemmed_sentence)

    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = []
    for i in preProcessed:
        if i not in final:
            if i.isdigit():
                pass
            else:
                if 'http' not in i:
                    final.append(i)
    temp1 = ' '.join(c for c in final)
    # print(preProcessed)
    return temp1


# Declaring secrets and tokens for our Twitter api
# consumer_key = 'DadKR3DKcG1PWvyh8igvAIaYN'
# consumer_secret = 'KjWoOAwm7uwwT0vTGWcuomuPq9Wglo5pA29kPxhOPvddMmO2Eg'
# access_token = '1266720191502680066-YGjG1jvAjIOOsG6NibYCDH7trAznfk'
# access_token_secret = '1E8fi5w2hi8eRyvnVvHcJnOU9p66oiJaXe5l1PAIQNYqA'
#
# auth = twp.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = twp.API(auth, wait_on_rate_limit=True)


# Using API call to get the tweets of the desired handle
def getdata(user):
    csvFile = open('Resource_Images/user.csv', 'w', newline='')
    csvWriter = csv.writer(csvFile)
    with fitz.open(user) as doc:
        pymupdf_text = ""
        for page in doc:
            pymupdf_text += page.get_text("text")
    data = pymupdf_text.split("\n")
    for i in data:
        csvWriter.writerow([i])
    csvFile.close()


def join(text):
    return "||| ".join(text)


# For fetching all the data from the specified handle
def twits(user):
    with fitz.open(user) as doc:
        pymupdf_text = ""
        for page in doc:
            pymupdf_text += page.get_text("text")
    data = pymupdf_text.split("\n")
    return data


# All the info for the processing is loaded. The data and frequency saved in their respective files are loaded.
# Vectorizer is defined and the models loaded.
# The model is fitted to provide the result and on the basis of result the personality is predicted.
# I/E, S/N, T/Fand P/J is chosen to get the personality. These letters are chosen on the basis of higher frequency.
def twit(handle):
    getdata(handle)
    with open('Resource_Images/user.csv', 'rt') as f:
        csvReader = csv.reader(f)
        data = [rows[0] for rows in csvReader]
    with open('Resource_Images/newfrequency300.csv', 'rt') as f:
        csvReader = csv.reader(f)
        mydict = {rows[1]: int(rows[0]) for rows in csvReader}

    vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
    x = vectorizer.fit_transform(data).toarray()
    df = pd.DataFrame(x)

    model_IE = pickle.load(open("Resource_Images/BNIEFinal.sav", 'rb'))
    model_SN = pickle.load(open("Resource_Images/BNSNFinal.sav", 'rb'))
    model_TF = pickle.load(open('Resource_Images/BNTFFinal.sav', 'rb'))
    model_PJ = pickle.load(open('Resource_Images/BNPJFinal.sav', 'rb'))

    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)

    b = Counter(IE)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("I")
    else:
        answer.append("E")

    b = Counter(SN)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("S")
    else:
        answer.append("N")

    b = Counter(TF)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("T")
    else:
        answer.append("F")

    b = Counter(PJ)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("P")
    else:
        answer.append("J")
    mbti = "".join(answer)
    return mbti


def split(text):
    return [char for char in text]


# All the characters are mapped to the respective personality detected
List_ch_I = ['Reflective',
             'Self-aware',
             'Take time making decisions',
             'Feel comfortable being alone',
             'Dont like group works']

List_ch_E = ['Enjoy social settings',
             'Do not like or need a lot of alone time',
             'Thrive around people',
             'Outgoing and optimistic',
             'Prefer to talk out problem or questions']

List_ch_N = ['Listen to and obey their inner voice',
             'Pay attention to their inner dreams',
             'Typically optimistic souls',
             'Strong sense of purpose',
             'Closely observe their surroundings']

List_ch_S = ['Remember events as snapshots of what actually happened',
             'Solve problems by working through facts',
             'Programmatic',
             'Start with facts and then form a big picture',
             'Trust experience first and trust words and symbols less',
             'Sometimes pay so much attention to facts, either present or past, that miss new possibilities']

List_ch_F = ['Decides with heart',
             'Dislikes conflict',
             'Passionate',
             'Driven by emotion',
             'Gentle',
             'Easily hurt',
             'Empathetic',
             'Caring of others']

List_ch_T = ['Logical',
             'Objective',
             'Decides with head',
             'Wants truth',
             'Rational',
             'Impersonal',
             'Critical',
             'Firm with people']

List_ch_J = ['Self-disciplined',
             'Decisive',
             'Structured',
             'Organized',
             'Responsive',
             'Fastidious',
             'Create short and long-term plans',
             'Make a list of things to do',
             'Schedule things in advance',
             'Form and express judgments',
             'Bring closure to an issue so that we can move on']

List_ch_P = ['Relaxed',
             'Adaptable',
             'Non judgemental',
             'Carefree',
             'Creative',
             'Curious',
             'Postpone decisions to see what other options are available',
             'Act spontaneously',
             'Decide what to do as we do it, rather than forming a plan ahead of time',
             'Do things at the last minute']


# Joins and returns the list of characters specific to the personality detected.
def character(text):
    o = split(text)
    characteristics = []
    for i in range(0, 4):
        if o[i] == 'I':
            characteristics.append('\n'.join(List_ch_I))
        if o[i] == 'E':
            characteristics.append('\n'.join(List_ch_E))
        if o[i] == 'N':
            characteristics.append('\n'.join(List_ch_N))
        if o[i] == 'S':
            characteristics.append('\n'.join(List_ch_S))
        if o[i] == 'F':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'T':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'J':
            characteristics.append('\n'.join(List_ch_J))
        if o[i] == 'P':
            characteristics.append('\n'.join(List_ch_P))
    char = '\n'.join(characteristics)
    data = char.split("\n")
    return data


# For making it into an exe:
# def resource_path(relative_path):
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")
#
#     return os.path.join(base_path, relative_path)


# Creating the Tkinter frontend for proper interaction


class MyWindow:
    def __init__(self, win):
        self.i = None
        self.e = None
        self.f = None
        self.s = None
        self.n = None
        self.tt = None
        self.j = None
        self.p = None

        self.t = None
        self.t1 = None
        self.t2 = None
        self.t3 = None

        self.b1 = None
        self.D_b1 = None
        self.D_btn1 = None

        self.lbl1 = None
        self.lbl2 = None
        self.lbl3 = None

        self.bg1 = None

        self.D_lbl0 = ttk.Label(win, text='Personality Prediction from CV / Portfolio', font=("Arial", 12))
        self.D_lbl0.place(x=55, y=30)
        self.btn1 = ttk.Button(win, text='Start Application', style='Accent.TButton', command=self.home)
        self.btn1.place(x=130, y=120)
        self.btn1 = ttk.Button(win, text='Quit', style='Accent.TButton', command=win.destroy)
        self.btn1.place(x=142, y=165)

    def home(self):
        win1 = Toplevel(window)
        win1.geometry("600x600")
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/Home.png")
        canvas = Canvas(win1, width=50, height=60)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        canvas.create_image(500, 70, image=self.bg1, anchor="ne")
        self.D_lbl0 = ttk.Label(win1, text='Personality Prediction from CV / Portfolio ',
                                font=("Arial", 15))
        self.D_lbl0.place(x=125, y=30)
        self.btn1 = ttk.Button(win1, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), win1.destroy()])
        self.btn1.place(x=150, y=120)
        self.btn1 = ttk.Button(win1, text='PREDICTION', style='Accent.TButton',
                               command=lambda: [self.linkedin(), win1.destroy()])
        self.btn1.place(x=350, y=120)

    def linkedin(self):
        win2 = Toplevel(window)
        win2.geometry("1280x720")
        self.D_lbl0 = ttk.Label(win2, text='Personality Prediction from CV / Portfolio',
                                font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win2, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win2.destroy()])
        self.btn1.place(x=455, y=120)
        self.btn1 = ttk.Button(win2, text='YOUR DATA', style='Accent.TButton',
                               command=lambda: [self.data(), win2.destroy()])
        self.btn1.place(x=565, y=120)
        self.btn1 = ttk.Button(win2, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [self.pp(), win2.destroy()])
        self.btn1.place(x=680, y=120)

    def data(self):
        win3 = Toplevel(window)
        win3.geometry("1280x720")
        self.D_lbl0 = ttk.Label(win3, text='Personality Prediction from CV / Portfolio',
                                font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win3, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win3.destroy()])
        self.btn1.place(x=455, y=120)
        self.D_btn1 = ttk.Button(win3, text='YOUR DATA', style='Accent.TButton',
                                 command=lambda: [win3.destroy(), self.data()])
        self.D_btn1.place(x=565, y=120)
        self.D_b1 = ttk.Button(win3, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [self.pp(), win3.destroy()])
        self.D_b1.place(x=680, y=120)
        self.t1 = Text(win3)
        self.t3 = Text(win3)
        self.t2 = ttk.Entry(win3)
        self.lbl1 = ttk.Label(win3, text='Enter file name: ')
        self.lbl1.place(x=490, y=180)
        self.lbl3 = ttk.Label(win3, text='Raw data of user:')
        self.lbl3.place(x=50, y=290)
        self.lbl3 = ttk.Label(win3, text='Cleaned data:')
        self.lbl3.place(x=680, y=290)
        self.t1.place(x=50, y=320)
        self.t3.place(x=680, y=320)
        self.t2.place(x=600, y=170, height=45)
        self.b1 = ttk.Button(win3, text='Get Data', style='Accent.TButton', command=self.twt)
        self.b1.place(x=250, y=240, width=130, height=50)
        self.b1 = ttk.Button(win3, text='Pre Process Data', style='Accent.TButton', command=self.twt1)
        self.b1.place(x=850, y=240, width=170, height=50)

    def twt(self):
        handle = self.t2.get()
        res = twits(handle)
        self.t1.configure(state='normal')
        self.t1.delete('1.0', END)

        self.t1.insert(END, str(res))
        self.t1.configure(state='disabled')

    def twt1(self):
        handle = self.t2.get()
        res1 = twits(handle)
        tx1 = join(res1)
        tx2 = pre_process(tx1)
        self.t3.configure(state='normal')
        self.t3.delete('1.0', END)

        self.t3.insert(END, str(tx2))
        self.t3.configure(state='disabled')

    def pp(self):
        win4 = Toplevel(window)
        win4.geometry("1280x720")
        self.D_lbl0 = ttk.Label(win4, text='Personality Prediction from CV / Portfolio',
                                font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win4, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win4.destroy()])
        self.btn1.place(x=455, y=120)
        self.D_btn1 = ttk.Button(win4, text='YOUR DATA', style='Accent.TButton',
                                 command=lambda: [self.data(), win4.destroy()])
        self.D_btn1.place(x=565, y=120)
        self.D_b1 = ttk.Button(win4, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [win4.destroy(), self.pp()])
        self.D_b1.place(x=680, y=120)
        self.lbl2 = ttk.Label(win4, text='Characteristics of Personalities:')
        self.lbl2.place(x=650, y=250)
        self.t = Text(win4, height=15, width=85)
        self.t.place(x=650, y=310)
        self.lbl2 = ttk.Label(win4, text='Predicted Personality Type')
        self.lbl2.place(x=200, y=437)
        self.lbl1 = ttk.Label(win4, text='Enter the file name(.pdf):')
        self.lbl1.place(x=150, y=307)
        self.t1 = ttk.Entry(win4)
        self.t1.place(x=350, y=300, height=40)
        self.t2 = Text(win4, width=10)
        self.t2.place(x=380, y=430, height=40)
        self.b1 = ttk.Button(win4, text='Predict Personality', style='Accent.TButton', command=self.predict)
        self.b1.place(x=300, y=375)

        self.i = ttk.Label(win4, text='I - Introvert')
        self.i.place(x=100, y=550)
        self.e = ttk.Label(win4, text='E - Extrovert')
        self.e.place(x=200, y=550)
        self.n = ttk.Label(win4, text='N - Intuitive')
        self.n.place(x=100, y=570)
        self.s = ttk.Label(win4, text='S - Sensing')
        self.s.place(x=200, y=570)
        self.f = ttk.Label(win4, text='F - Feeling')
        self.f.place(x=100, y=590)
        self.tt = ttk.Label(win4, text='T - Thinking')
        self.tt.place(x=200, y=590)
        self.j = ttk.Label(win4, text='J - Judging')
        self.j.place(x=100, y=610)
        self.p = ttk.Label(win4, text='P - Perceiving')
        self.p.place(x=200, y=610)

    def predict(self):
        handle = self.t1.get
        res = twit(handle)
        self.t2.configure(state='normal')
        self.t2.delete('1.0', END)

        self.t2.insert(END, str(res))
        self.t2.configure(state='disabled')
        r = self.t2.get
        result = character(res)
        self.t.configure(state='normal')
        self.t.delete('1.0', END)

        for i in range(len(result)):
            self.t.insert(END, str(result[i]))
            self.t.insert(END, str('\n'))
        self.t.configure(state='disabled')

    def mbti(self):
        win5 = Toplevel(window)
        win5.geometry("1280x720")
        self.D_lbl0 = ttk.Label(win5, text='Personality Prediction from CV / Portfolio', font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win5, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win5.destroy()])
        self.btn1.place(x=420, y=120)
        self.btn1 = ttk.Button(win5, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [win5.destroy(), self.mbti()])
        self.btn1.place(x=520, y=120)
        self.btn1 = ttk.Button(win5, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), win5.destroy()])
        self.btn1.place(x=620, y=120)
        self.btn1 = ttk.Button(win5, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [self.explore(), win5.destroy()])
        self.btn1.place(x=720, y=120)
        canvas = Canvas(win5, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/data_info.png")
        canvas.create_image(1050, 70, image=self.bg1, anchor="ne")

    def mbt(self):
        win6 = Toplevel(window)
        win6.geometry("1280x720")
        self.D_lbl0 = ttk.Label(win6, text='Personality Prediction from CV / Portfolio',
                                font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win6, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win6.destroy()])
        self.btn1.place(x=420, y=120)
        self.btn1 = ttk.Button(win6, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), win6.destroy()])
        self.btn1.place(x=520, y=120)
        self.btn1 = ttk.Button(win6, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [win6.destroy(), self.mbt()])
        self.btn1.place(x=620, y=120)
        self.btn1 = ttk.Button(win6, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [self.explore(), win6.destroy()])
        self.btn1.place(x=720, y=120)
        canvas = Canvas(win6, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/TestResults.png")
        canvas.create_image(1050, 70, image=self.bg1, anchor="ne")

    def explore_fn(self, win, canvas, bg1):
        canvas.create_image(50, 70, image=bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(win, text='Personality Prediction from CV / Portfolio',
                                font=("Arial", 20))
        self.D_lbl0.place(x=405, y=30)
        self.btn1 = ttk.Button(win, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home(), win.destroy()])
        self.btn1.place(x=420, y=120)
        self.btn1 = ttk.Button(win, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), win.destroy()])
        self.btn1.place(x=520, y=120)
        self.btn1 = ttk.Button(win, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), win.destroy()])
        self.btn1.place(x=620, y=120)
        self.btn1 = ttk.Button(win, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [win.destroy(), self.explore()])
        self.btn1.place(x=720, y=120)
        self.btn1 = ttk.Button(win, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), win.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(win, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), win.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(win, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), win.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(win, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), win.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(win, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), win.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(win, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), win.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore(self):
        win7 = Toplevel(window)
        win7.geometry("1280x720")
        canvas = Canvas(win7, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/CountPlot.png")
        self.explore_fn(win7, canvas, self.bg1)

    def explore1(self):
        win8 = Toplevel(window)
        win8.geometry("1280x720")
        canvas = Canvas(win8, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/PiePlot.png")
        self.explore_fn(win8, canvas, self.bg1)

    def explore2(self):
        win9 = Toplevel(window)
        win9.geometry("1280x720")
        canvas = Canvas(win9, width=2500, height=2000)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/DisPlot.png")
        self.explore_fn(win9, canvas, self.bg1)

    def explore3(self):
        win10 = Toplevel(window)
        win10.geometry("1280x720")
        canvas = Canvas(win10, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/I_E.png")
        self.explore_fn(win10, canvas, self.bg1)

    def explore4(self):
        win11 = Toplevel(window)
        win11.geometry("1280x720")
        canvas = Canvas(win11, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/N_S.png")
        self.explore_fn(win11, canvas, self.bg1)

    def explore5(self):
        win12 = Toplevel(window)
        win12.geometry("1280x720")
        canvas = Canvas(win12, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/T_F.png")
        self.explore_fn(win12, canvas, self.bg1)

    def explore6(self):
        win13 = Toplevel(window)
        win13.geometry("1280x720")
        canvas = Canvas(win13, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file="Resource_Images/J_P.png")
        self.explore_fn(win13, canvas, self.bg1)


window = tk.Tk()
window.tk.call('source', 'Resource_Images/forest-dark.tcl')

# Set the theme with the theme_use method
ttk.Style().theme_use('forest-dark')

window.title("GAMINE")
my_win = MyWindow(window)
window.geometry("400x300")
window.iconphoto(True, tk.PhotoImage(file='Resource_Images/icon.png'))
window.mainloop()
