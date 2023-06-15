###################################################
# Sorting Products ( Ürün Sıralama)
# Bu bölümde herhangi bir ürün, ilan, başvuru vs gibi durumları sıralama yaparken neye göre sıralama yapacağımızı hangi
# kriter veya kriterlerin daha ağır basacağı istenen şartların ağrılıklarıın neler olacağını belirlemeyi öğreneceğiz.
###################################################


# Bazı faktörlere göre, bazı ürünleri yada kişileri yada odaklanılan her hangi bir nesneyi sıralama ihtiyacımız
# olmaktadır.
# Biz bu uygulamada e ticaret tarafındaki ürün sıralama problemine çözüm arayacağız. Bazı sıralama tekniklerine bakış
# açısı kazanacağız.
# Daha sonra istatistiksel bir teknik ile bu çözümlerin bilimsel halinin nasıl olduğunu öğreneceğiz.


###################################################
# Uygulama: Kurs Sıralama
###################################################

# burada gerekli kütüphaneler programa yüklendi.
import pandas as pd
import math # bayesian için
import scipy.stats as st # bayesian için
from sklearn.preprocessing import MinMaxScaler # bu kütüphaneyi standartlaştırma işelmi belirli bir aralığa getirmek
# için programa dahil ettik.

# satır sütunlarda görünürlük ayarları yapıldı.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Miuul/Ölçüm Problemleri/datasets/product_sorting.csv")
print(df.shape)
df.head(10)

####################
# Sorting by Rating (Derecelendirmeye Göre Sıralama)
####################

df.sort_values("rating", ascending=False).head(20) # burada df i rating e göre büyükten küçüğe göre sıralama yaptık.

####################
# Sorting by Comment Count or Purchase Count ( Yorum ve Satın alma Sayısına Göre Sıralama)
####################

df.sort_values("purchase_count", ascending=False).head(20) # burada df i satın alma sayısına göre büyükten küçüğe doğru
# sıralam yaptık
df.sort_values("commment_count", ascending=False).head(20) # burada yorum sayısına göre df i büyükten küçüğe sıralama
# yaptık.

####################
# Sorting by Rating, Comment and Purchase (Derecelendirme, Satın Alma, Yorum Sayısına Göre Sıralama)

# Burada şunu yapacağız: data frame içersinde yer alan her üç durumu da sıralama yaparken dikkate alacağız. fakat burada
# şöyle bir sıkıntı var. Bu değişkenlerin değerleri aralıkları çok alakalı bir aralıkta olmadığı için bunları aynı
# aralığa ve standarta getirdikten sonra sıralma yapacağız.
####################

# aşağıdaki kodda şunu yaptık satın alma sayısını (purchase_coount) MinMaxSacaler fonksiyonu ile 1 ile 5 aradında
# değerelere indirgedik.
# böylelikle satın alma sayısı ile rating aynı aralığa yani karşılaştırabilir bir standarda gelmiş oldu.
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

# yukarıda ki gibi yorum sayısını da satın alma sayısı gibi 1 ile 5 arasına indirgedik. böylelikle her üç değişken
# raitng purcahse_count ve comment_count sayıları aynı aralığa indirgenmiş oldu.
# böylelikle her üç değişken de aynı standarda gelmiş oldu.
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

# yukarıda belirli bir standarda getirdiğimiz değerleri  artık istediğimiz ağırlıkları vererek her bir kurs için bir
# skor oluşturabiliriz.
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)
# burada elde ettiğimiz değer belirli bir çok faktörün ağırlığı ile elde dilen skorlardır. Rating değildir.

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df) # ağırlıklı sıralama skoru isminde yeni bir değişken
# oluşturdurk df in içersinde bu değişkene de  yazdığımız foknsiyondan elde edilen çıktılar gönderildi.

df.sort_values("weighted_sorting_score", ascending=False).head(20) # df ağılrıklı sıralma skoruna göre büyükten küçüğe
# göre sıraldık.

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)
# burada df içersinden kurs isimleri içersinde veri bilimi stringi geçen kursları ağırlıklı skora göre sıralama yaptık
# büyükten küçüğe doğru.

####################
# Bayesian Average Rating Score (Bayes Ortalama Derecelendirme Puanı)

# Bu konuda sıralama yaparken dikkat etmemiz gereken konulardan biri olan rating e bu konuda tekrar odaklanacağız.
# Bu odaklanma neticesinde bir skor elde edeceğiz ve bu skor bir ürünün ortalama nihai puanı olarak da kullanılabilir,
# skor olarak da değerlendirilebilir. Skor olarak değerlendirilmesi mantıklıdır.
####################

# Bu konu karşımıza aşağıdaki gibi gelebilir.
# Sorting Products with 5 Star Rated  => 5 yıldızlı sistemler de ürün sıralama
# Sorting Products According to Distribution of 5 Star Rating => 5 yıldızın dağılımına göre ürün sıralama şeklinde...

# bizim odağımız bu puanarın dağılımı. Bu puanların dağılım bilgisini kullanarak olasılıksal bir ortalama
# hesaplayacağız

#  5_point  4_point  3_point  2_point  1_poin
#  3466      924      185       46        6
#  3466      924      185       46        6
#  ....      ....     ....      ...       ...
# Bayesian_average_rating : Puan dağılımlarının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar. bu
# puanların dağılımı üzerinden bir ortalama hesabı yapacağız.

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

# yukarıda ki fonksiyonda bayesian ile ortalama hesaplamak için kullanacağız. bu fonksiyonda önemli olan n değeridir.
# n değeri girilecek olan yızdızların ve bu yıldızların gözlenme frekanslarını ifade etmektedir.
# confidance değeri güven değeridir. genelde 0.95 veya 0.99 alınır. ama genelde 0.95 alınır.
# n 5 elemanlı:
# 1. elemanında 1 yıldızdan kaç tane var
# 2. elemanında 2 yıldızdan kaçtane var
# 3. elemanında 3 yıldızdan kaç tane var
# 4. elemanında 4 yıldızdan kaçtane var
# 5. elemanında 5 yıldızdan kaç tane var

# Not:Bar score bize sadece ratinglere odaklanarak sıralama sağladı.
df.head()

# aşağıda ki kodda yularıda bahsettiğimiz n değerini bayesian fonksiyonunua yolladık hesaplama yapmak için.
# bar_score isminde bir değişken oluşturup bayesian fonksiyonundan elde ettiğimiz değerleri de bu değişkene atadık.
# bayesian bar_scor ifadesi ile de karşımıza çıkabilir.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating => yukarıda ki iki ortalamayı kullanarak ağırlıklı ortalama hesapladık
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler

# aşağıda şunu yaptık: daha önce hesapladığımız ağırlıklı ortalama (wss) ile bayesian (bar_scor) bir araya getirip tüm
# koşulları bir arada değerlendirip bir skor elde edeceğiz.

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df) # yukarıda ki fonksiyonda elde ettiğimiz sonuçalrı df içersine
# yeni bir değişken oluşturarak ekledik.

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)