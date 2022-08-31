import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import csv


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def pdf_kwd(file):
    from textract import process
    from nltk.tokenize import word_tokenize

    text = process(file, method='tesseract', encoding='utf_8', language='por')
    text = text.decode('utf-8').replace('\\n', ' ').replace('\n', ' ')

    token = word_tokenize(str(text))
    token = [word.lower() for word in map(lambda x: x.lower(), token)]

    keywords = [word for word in map(lambda x: x.lower(), token)]
    keywords = ' '.join(keywords)
    return keywords


# List of Candidates to analyse
candidates = ['soraya', 'simone', 'lula', 'bolsonaro', 'felipe', 'ciro']

# Read files
plan = {}

for x in candidates:
    plan.update({x: pdf_kwd('planos/'+x+'.pdf')})

# Words to drop
stop_words = []
with open('stop_words.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        stop_words += row


# Mask for WordCloud
mask = np.array(Image.open('brazil.png'))

for i, x in enumerate(candidates):
    text = plan[x]
    cloud = WordCloud(background_color='white', stopwords=stop_words,
                      max_words=2000,
                      mask=mask, contour_width=0).generate(text)
    plt.figure()
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(x.capitalize())
    plt.axis('off')
    plt.savefig('imagens/'+x+'.png', dpi=1200,
                bbox_inches='tight', transparent=True)
