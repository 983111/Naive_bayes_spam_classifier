import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(counter, title):

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(counter)

    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    plt.show()