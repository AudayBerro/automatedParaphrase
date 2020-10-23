from nltk.translate.chrf_score import sentence_chrf

ref = 'the cat is on the mat'.split()
hyp = 'the the the the the the'.split()
a = sentence_chrf(ref, hyp)
print(a)