# We can generate our results in the form of WordClouds
list_of_2grams = []
list_of_2grams.extend([i for i in df_english['2grams']])
list_of_3grams = []
list_of_2grams.extend([i for i in df_english['3grams']])

flat_list_of_2grams = [item for sublist in list_of_2grams for item in sublist]
flat_list_of_3grams = [item for sublist in list_of_3grams for item in sublist]

strings_of_2grams = ' '.join(g for g in flat_list_of_2grams)
wordcloudimage_2grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_2grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_2grams, interpolation='bilinear')
plt.axis("off")
plt.show()

strings_of_3grams = ' '.join(g for g in flat_list_of_3grams)
wordcloudimage_3grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_3grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_3grams, interpolation='bilinear')
plt.axis("off")
plt.show()

frequency_2grams = nltk.FreqDist(list_of_2grams)
plt.figure(figsize=(29,10))
frequency_2grams.plot(150,cumulative=False)

frequency_3grams = nltk.FreqDist(list_of_3grams)
plt.figure(figsize=(29,10))
frequency_3grams.plot(150,cumulative=False)
