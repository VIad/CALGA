
# CALGA (Cyrillic Attention-based Lyrics Generation Algorithm)

This is not science, this is a mid-effort shitpost
## Abstract

Lyrics generation is an intriguing challenge in natural language processing, especially when applied to the unique context of Cyrillic script. In this article, we introduce the Cyrillic Attention-Based Lyrics Generation Algorithm (CALGA) and present a comparative study of various transformer-based architectures for this specific task.

Our investigation focuses on evaluating the quality of generated lyrics and the fluency of the output text, using perplexity (PPL) as the primary metric. By comparing multiple transformer architectures, we shed light on the nuances and performance variations among these models in the context of Cyrillic lyrics generation.

The results of our evaluation offer valuable insights into the strengths and weaknesses of different transformer architectures when applied to creative text generation. This research contributes to our understanding of the state-of-the-art in lyrics generation and provides guidance for selecting the most suitable transformer-based model for specific creative and linguistic applications, advancing the field of natural language generation.


## Introduction
This project is inspired by [@Kyosek/text-generator-Krisko](https://github.com/kyosek/text-generator-Krisko)

Popfolk, a distinctive and influential music genre in Bulgaria, has captivated audiences with its unique blend of traditional Bulgarian folk elements and modern pop sensibilities. Rooted in the rich cultural heritage of Bulgaria, popfolk has evolved into a dynamic and popular genre over the years, celebrated for its fusion of traditional melodies with contemporary styles. In this introductory exploration, we delve into the world of Bulgarian popfolk, its historical context, famous artists, and its enduring impact on the country's music scene.

It is futile to attempt to statistically model what takes great creativity and lyrical craftsmanship. The task of generating coherent Bulgarian pop-folk lyrics is incredibly complex, mainly because of their varying rhyme structures (**ABABBCBC**), **(BCBC)**
and because of the numerous Phraseme and genre-unique word semantics.

A few examples are:
Щибиди доп доп
Щрак
рак так так
## The Plan
Kyosek utilizes a [Bidirectional Long Short term memory architecture](https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/) for his Krisko lyrics generations. He generates single sentences.

In this project, we'll utilize the power of the newer [Transformer architecture](https://arxiv.org/abs/1706.03762). Addditionally, we'll attempt to generate full-length songs, in hopes of trying to capture and model some of the rhyming traditionally found in Bulgarian Pop-folk.

To do that, we need a couple of things first
1. The dataset
2. Exploratory data analysis of the corpus
3. Data augmentation & sanitisation

## The Dataset
There are a lot of bulgarian Pop-folk songs. Approximately **5300** as of the time of writing this. This fact alone makes our task much simpler than Kyosek's as we have access to a much bigger dataset.

We use BeautifulSoup to scrape [Textove.org](https://www.tekstove.org), a website storing thousands of bulgarian songs. **We only scrape the Pop-folk category**.

## EDA
We get all songs and analyze the 15 most frequently occuring artists:
![enter image description here](https://i.imgur.com/8TLyQCN.png)

In Bulgarian pop-folk, duets are common, (>10% of songs).
Therefore it is interesting to know how many songs are duets, trios, quartets.. etc:
![enter image description here](https://i.imgur.com/ZYvlo0n.png)

The most commonly occuring words in Bulgarian pop-folk (minus [stopwords](https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/)) are:
![enter image description here](https://i.imgur.com/Mo6bphf.png)
![enter image description here](https://i.imgur.com/4IEiHSs.png)

Histogram, plotting the frequency of song lengths
![enter image description here](https://i.imgur.com/ATmk6EB.png)

An unprocessed training example:

колко рани си оставил в мене 
няма място - виждаш ли?
ако почна сега да броя 
колко пъти съм сама.. нямат край!
а всеки път се лъжа отново
и всеки път боли, спри!
ти признаваш ли? - не, не, не, не!
и колкото те мразя,
не мога с друг - нее!
сърцето си не пазя 
и мойто тяло - нее!
няма да ти кажа "не"
лесно ми разбъркваш мислите..
с дрехите пободно е,
зная - махаш ги добре!
колко пъти си забравял за мене 
как се чувствам, питаш ли?
колко нощи преспивам страха,
че не си ме пожелал.. нямат край!
а всеки път се лъжа отново
и всеки път боли, спри!
ти признаваш ли? - не, не, не, не!
и колкото те мразя,
не мога с друг - нее!
сърцето си не пазя 
и мойто тяло - нее!
няма да ти кажа "не"
лесно ми разбъркваш мислите..
с дрехите пободно е,
зная - махаш ги добре!
## Preprocessing & data sanitization / augmentation
The scraped data was passed through the following pipeline

 1. Split data into two sets - Train and CV *(<10%)*
 2. Lowercase all text
 3. Split songs and remove useless punctuation (eg (x2), : /4)
 4. Place each song in a single line while replacing \n with special "nl" character
 5. Prefix each song with $ token which will be used as start of sequence


Additionally, before modelling with the corpus, the data was [n-gramified](https://en.wikipedia.org/wiki/N-gram) so that the corpus is greatly augmented. 

This means that if we have a sentence like 
**The quick brown fox jumps over the lazy dog**

it will be n-gramified like so:

*The quick brown fox jumps over*
*quick brown fox jumps over the*
*brown fox jumps over the lazy*
*fox jumps over the lazy dog*



## Modelling
As mentioned above, we'll be using the Transformer architecture to generate our lyrics.

### A few words on the architecture
![enter image description here](https://i.imgur.com/nmFRgSo.png)

A noteworthy thing to mention about the transformer architecture is that it was initially **designed for machine translation** i.e for sequence to sequence tasks. That's why the architecture above has two parts.
The left part is the transformer encoder and the right part is the transformer decoder.

The architecture uses attention, and the values from the encoder are fed into the decoder via the equation for **cross-attention**:

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
In this equation, the **Q** and **K** (Queries and Key) values come from the encoder and decoder, respectively.

In our case, we won't be translating sequences, only generating them and therefore will only use the decoder part of the transformer.

So, instead of cross-attention, we'll utilize **self-attention**. We'll pass the **Q** value again, instead of a **K** value, so the algorithm will 'pay attention' to the already generated words when generating
$$ P(w_i  | w_1..w_{i-1}) $$ 
The equation for a self-attention head effectively becomes
$$ SelfAttention(Q, Q, V) = softmax(\frac{QQ^T}{\sqrt{d_k}})V $$

And the Multi-headed attention
$$ MultiHead(Q, Q, V ) = Concat(head_1, ..., head_h)W_O $$

### Different models, tested

For this specific task, i opted to training a transformer model from the ground up, instead of fine-tuning an existing LLM or doing transfer learning.
The reason for this is mainly lyrical and genre authenticity as well as the lack of cloud GPU funding raised for this crucial NLP project.

During my model comparisons, i found that the sweet spot is 3 transformer blocks, less fail to generate even semi-coherent bulgarian and more lead to diminishing returns or even worse performance.
For the vocabulary & tokenization, i use google [WordPiece](https://paperswithcode.com/method/wordpiece). The embeddings were also custom trained with varying size embedding vectors.

The model architecture is therefore:

![enter image description here](https://i.imgur.com/vWQYKbF.png)

All models were trained up to **25 epochs**, using the **ADAM** optimizer.
The first model was trained to only generate verses, whereas all other models were trained to generate full songs. All models have a FC (fully connected dense) layer size of 2048.




| Model | Attention heads | Embedding dim | Context size | Hidden state dim | Dropout rates for blocks 1, 2, 3 |
|-------|-----------------|---------------|--------------|------------------|----------------------------------|
| V1    | 12              | 256           | 128          | 21               | 0.6 \| 0.7 \| 0.7                |
| V2    | 24              | 256           | 256          | 10               | 0.6 \| 0.7 \| 0.7                |
| V3    | 24              | 300           | 256          | 12               | 0.6 \| 0.6 \| 0.7                |
| V4    | 4               | 256           | 256          | 64               | 0.6 \| 0.6 \| 0.7                |
| V5    | 5               | 280           | 256          | 56               | 0.6 \| 0.6 \| 0.7                |

Models are evaluated using [Perplexity (PPL)](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72) which is computed on both the training set and the cross-validation set.
![enter image description here](https://i.imgur.com/lJcTjED.png)

By default, the transformer uses a vector of size dk for the **Q** and **V** values, determined as follows
$$ d_k = d_v = \frac{d_{embeddings}}{h}  $$
where h is the amount of attention heads. I've found that using too many attention heads (*models 2 & 3*) generally yields worse results, due to the bottleneck created by the small hidden state vector. I've found that the sweet spot for 
$$ d_{embeddings} \in [256; 300] $$
and for
$$ h \in [4;5] $$

After the final FC layer, i use [Top P sampling](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=lab-prompt-parameters) for the next token generation. I found that with $$ p \in [0.80; 0.85]$$ i get the funniest / most creative generations.


Top P works by finding the least amount of tokens, such that
$$ \sum_i{P(w_i |context) > p} $$

and then samples from the new token distribution
## Some generations

като на любов играеш си как мирише на твойта кожа, 
 и тя дори и да сме с теб, 
 без да криеш любовта ни пред хората 
 и аз да спирам рано в седем дни. 
 няма да те имам. 
 да съм запукам в клевикаш няма скука, от купони ще те гукам 
 бръмбарите в моята глава полудяха 
 ще ти правя пеперуди във стомаха 
 ако трябва ще те спирам рано в седем дни 
 ще те спирам рано в седем дни 
 обещавам да не спирам рано в седем нощи аз 
 с вкус на мода 
 бръмбарите в морето луната като ги омпителната жертва ти ми даваш любовта ни. 
 ще те взимам навсякъде и ще говоря 
 ще ти спирам рано в седем дни. 
 ще те целувам, ще те спирам рано в седем нощи специпчата и ще те бръмбарите в твойта глава полудяха 
 ще ти правя пеперуди във стомаха 
 всичките пари вътре 

-------------------
в мен какво би? 
 имам всичко да ти дам да ти дам да ме слеят две сърца. 
 всеки миг живот боли, 
 но едва. 
 но какво сега от това, 
 как да вярвам, 
 боли да ме боли 
 дано обичаш да страдаш ти, 
 дано щастлива с него да станеш луда 
 дано обичаш да страдаш ти. 
 дано на теб да се превърна, 
 дано на мен да си плащаш за греха. 
 дано на теб да ти дам да се влюбиш дано на мен 
 дано на тебе да се влюбиш дано на тебе да мечтаеш. 
 но да забравиш, да ама с мене. 
 дано на тебе да полудеем, 
 дано на тебе да съм щастлива. 
 дано на тебе да боли, 
 дано на тебе да останеш 
 дано на тебе да страдаш ти, 
 дано на тебе да обичаш да страдаш ти? 
 дано на тебе да видиш ти. 
 дано на тебе да

-----
 как ли ти стои? 
 ето ме да ме спреш да ме спреш. 
 с тебе тръгвам навсякъде да видя? 
 всичко мое, искам, искам всичко да е до мен 
 и как го направя? 
 защо да живеем двама, 
 сякаш просто няма да сме заедно? 
 всичко води към мене. 
 пия, скитам се да избягам, но накъде. 
 но ще избягам, аз не мога да избягам, не мога да избягам. 
 тръгвам си от теб 
 нищо друго не казвай 
 че ме лъжеш ти. 
 без тебе тръгвам си да избягам, но накъде 
 обичам да избягам, там където искаш да се сбогувам 
 не си с друг живот да се сбогувам 
 любовта да избягам, знам това е.


----
зарежда ти. 
 какво ми става? 
 най-добрия във града, 
 за секунда ако можеш, с мене си сега. 
 и секунда ако хвана с пари. 
 готов съм, с мене, 
 а после пак ще останеш за любов. 
 да те отпробваме! 
 щом за тази вечер поршето? 
 щом отново си от тогава 
 все едно не прощава, 
 дори не се преборотен 
 твоя съм сега, 
 ще ми го чуеш танцувай с мен, 
 нали ме сваляш, а всеки път тежки стъпки чуя, 
 аз виждам, а да видя бърза, то, 
 не ме вълнуваш, 
 сега до края, 
 и с нея ти не се преборих. 
 когато съмнения прошепвай, 
 но сърцето ми, 
 колкото си тръгнеш, 
 всеки ден, всяка нощ. 
 дай ръка, където 
 да си до мене, 
 тогава се изпрати, 
 в съня, щом си на нея.

## Conclusion
This was a fun little side project for me, used mainly to learn about the transformer architecture. There will be years before even the biggest LLMs are able to nearly capture the semantics, rhymes and lyricism that is present in Bulgarian pop-folk. These small models explored here certainly fail to do so.

What they don't fail in however is generating funny semi-coherent song lyrics, in which even rhyming can be seen.

