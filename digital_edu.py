#создай здесь свой индивидуальный проект!
import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv('train.csv')
# df.info()
# print(df['sex'].value_counts())
# men = 0
# women = 0
# male_students = 0
# female_students = 0
# def sex_count(row):
#     global men, women, male_students, female_students
#     if row['sex'] == 1:
#         women += 1
#         if row['result'] == 1:
#             female_students += 1
#     if row['sex'] == 2:
#         men += 1
#         if row['result'] == 1:
#             male_students += 1
# df.apply(sex_count, axis = 1)
# studingmen_procent = male_students/men
# notstudingmen_procent = (men-male_students)/men
# s = pd.Series(data = [studingmen_procent, notstudingmen_procent], index = ['мужчины которые купили курс', 'мужчины которые не купили курс'])
# s.plot(kind='pie')
# plt.show()
# studingwomen_procent = female_students/women
# notstudingwomen_procent = (women-female_students)/women
# s = pd.Series(data = [studingwomen_procent, notstudingwomen_procent], index = ['женщины которые купили курс', 'женщины которые не купили курс'])
# s.plot(kind='pie')
# plt.show()
# men_procent = male_students/(men+women)
# women_procent = female_students/(men+women)
# s = pd.Series(data = [men_procent, women_procent], index = ['Занимающиеся мужчины', 'Занимающиеся женщины'])
# s.plot(kind='pie')
# plt.show()
# print(df['occupation_type'].value_counts())
# workers = 0
# university = 0
# workers_students = 0
# university_students = 0
# def occupation_count(row):
#     global workers, university, workers_students, university_students
#     if row['occupation_type'] == 'university':
#         university += 1
#         if row['result'] == 1:
#             university += 1
#     if row['occupation_type'] == 'work':
#         workers += 1
#         if row['result'] == 1:
#             workers_students += 1
# df.apply(occupation_count, axis = 1)
# studing_workers = workers_students/workers
# notstuding_workers = (workers - workers_students)/workers
# s = pd.Series(data = [studing_workers, notstuding_workers], index = ['Занимающиеся люди работающие на работе', 'Люди работающие на работе которые не занимаются'])
# s.plot(kind='pie')
# plt.show()
# studing_university = university_students/university
# notstuding_university = (university - university_students)/university
# s = pd.Series(data = [studing_university, notstuding_university], index = ['Занимающиеся студенты', 'студенты которые не занимаются'])
# s.plot(kind='pie')
# plt.show()
# university_procent = university_students/(workers_students + university_students)
# workers_procent = workers_students/(workers_students + university_students)
# s = pd.Series(data =[university_procent, workers_procent], index = ['Занимающиеся студенты', 'Занимающиеся люди которые работают на работе'])
# s.plot(kind='pie')
# plt.show()



















df2 = pd.read_csv('train.csv')
df2.drop(['id','bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'education_form', 'relation', 'education_status', 'langs', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)
df2.info()
df2['occupation_type'].fillna('university', inplace = True)
df2[list(pd.get_dummies(df2['occupation_type']).columns)] = pd.get_dummies(df2['occupation_type'])
df2.drop(['occupation_type'], axis = 1, inplace = True) 
df2.info()
def change2(row):
    if row['sex'] == 1:
        return 0
    elif row['sex'] == 2:
        return 1
df2['sex'] = df2.apply(change2, axis = 1)
print(df2['sex'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
X = df2.drop('result', axis = 1)
y = df2['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))