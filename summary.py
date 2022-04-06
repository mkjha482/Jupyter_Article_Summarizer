from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as nmp
import networkx as netx


def read_article():
    Article = "The COVID-19 pandemic has led to a dramatic loss of human life worldwide and presents an unprecedented challenge to public health, food systems and the world of work. The economic and social disruption caused by the pandemic is devastating: tens of millions of people are at risk of falling into extreme poverty, while the number of undernourished people, currently estimated at nearly 690 million, could increase by up to 132 million by the end of the year. Millions of enterprises face an existential threat. Nearly half of the world’s 3.3 billion global workforce are at risk of losing their livelihoods. Informal economy workers are particularly vulnerable because the majority lack social protection and access to quality health care and have lost access to productive assets. Without the means to earn an income during lockdowns, many are unable to feed themselves and their families. For most, no income means no food, or, at best, less food and less nutritious food. The pandemic has been affecting the entire food system and has laid bare its fragility. Border closures, trade restrictions and confinement measures have been preventing farmers from accessing markets, including for buying inputs and selling their produce, and agricultural workers from harvesting crops, thus disrupting domestic and international food supply chains and reducing access to healthy, safe and diverse diets. The pandemic has decimated jobs and placed millions of livelihoods at risk. As breadwinners lose jobs, fall ill and die, the food security and nutrition of millions of women and men are under threat, with those in low-income countries, particularly the most marginalized populations, which include small-scale farmers and indigenous peoples, being hardest hit. Millions of agricultural workers – waged and self-employed – while feeding the world, regularly face high levels of working poverty, malnutrition and poor health, and suffer from a lack of safety and labour protection as well as other types of abuse. With low and irregular incomes and a lack of social support, many of them are spurred to continue working, often in unsafe conditions, thus exposing themselves and their families to additional risks. Further, when experiencing income losses, they may resort to negative coping strategies, such as distress sale of assets, predatory loans or child labour. Migrant agricultural workers are particularly vulnerable, because they face risks in their transport, working and living conditions and struggle to access support measures put in place by governments. Guaranteeing the safety and health of all agri-food workers – from primary producers to those involved in food processing, transport and retail, including street food vendors – as well as better incomes and protection, will be critical to saving lives and protecting public health, people’s livelihoods and food security. In the COVID-19 crisis food security, public health, and employment and labour issues, in particular workers’ health and safety, converge. Adhering to workplace safety and health practices and ensuring access to decent work and the protection of labour rights in all industries will be crucial in addressing the human dimension of the crisis. Immediate and purposeful action to save lives and livelihoods should include extending social protection towards universal health coverage and income support for those most affected. These include workers in the informal economy and in poorly protected and low-paid jobs, including youth, older workers, and migrants. Particular attention must be paid to the situation of women, who are over-represented in low-paid jobs and care roles. Different forms of support are key, including cash transfers, child allowances and healthy school meals, shelter and food relief initiatives, support for employment retention and recovery, and financial relief for businesses, including micro, small and medium-sized enterprises. In designing and implementing such measures it is essential that governments work closely with employers and workers."
    sentences = []
    
    #splitting the provided article into sentences based on the parameter ". "
    sentences = Article.split(". ")
    
    return sentences


def sentence_similarity(wd1, wd2, stopwords = None):
    if stopwords is None:
        stopwords = []
    
    #lowercase conversion of the words
    wd1 = [x.lower() for x in wd1]
    wd2 = [x.lower() for x in wd2]
    
    #making a set to ensure there are no repetitions in the list
    all_words = list(set(wd1 + wd2))
    
    #vector creation for cosine comparison
    vector1 = [0]*(len(all_words))
    vector2 = [0]*(len(all_words))
    
    for i in wd1:
        if i in stopwords:
            continue
        vector1[all_words.index(i)] += 1
    
    for i in wd2:
        if i in stopwords:
            continue
        vector2[all_words.index(i)] += 1
    
    #returning the value after computing the cosine of the vector angles
    return 1 - cosine_distance(vector1, vector2)


def b_similarity_matrix(sentences, stop_words):
    #creating a null/empty matrix 
    similar_matrix = nmp.zeros((len(sentences), len(sentences)))
    
    for ind1 in range(len(sentences)):
        for ind2 in range(len(sentences)):
            #if two sentences are same, no need to generate a matrix as they are already similar, so, we skip matrix generation
            if(ind1 == ind2):
                continue
                
            #generating the similarity matrix for each sentence in the article    
            similar_matrix[ind1][ind2] = sentence_similarity(sentences[ind1], sentences[ind2], stop_words)

    return similar_matrix


def summary_generator(net = 4):
    stop_words = stopwords.words('english')
    summarize_article = []
    
    #calling function to read the article
    sentences = read_article()
    
    #function call to generate similarity matrix
    sentence_similarity_martix = b_similarity_matrix(sentences, stop_words)
    
    #seuential ranking based on similarity matrix
    sentence_similarity_graph = netx.from_numpy_array(sentence_similarity_martix)
    scores = netx.pagerank(sentence_similarity_graph)
    
    #rank sorting and sentence selection
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)
    
    #printing the picked sentences based on how many lines we are supposed to print as user input.
    for i in range(net):
      summarize_article.append(" ".join(ranked_sentence[i][1]))
    
    #printing the summary
    print(". ".join(summarize_article))
    

summary_generator()