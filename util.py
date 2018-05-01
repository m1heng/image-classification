def Accuracy(listA, listB):
	if len(listB) != len(listA):
		return 0
	matches = 0
	for x, y  in zip(listA,listB):
		if x == y:
			matches += 1
	return matches/len(listA)