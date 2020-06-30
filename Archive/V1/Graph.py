
class Graph:
    def __init__(self,rules):
        '''
        :param rules: list of Classifiers. Key is classifier index in rule population.
        '''
        self.rules = {}
        for classifierIndex in range(len(rules)):
            self.rules[classifierIndex] = Vertex(rules[classifierIndex],rules,classifierIndex)

    def addOutgoingEdgeFromTo(self,fromRuleIndex,toRuleIndex,weight):
        self.rules[fromRuleIndex].addOutgoingEdgeTo(toRuleIndex,weight)

    def getRelatedRules(self,ruleIndex):
        return self.rules[ruleIndex].getMostRelatedRules()

    def subtractFromAllEdges(self,amount=1):
        for rule in self.rules:
            self.rules[rule].substractFromAllEdges(amount)

    def getMaxWeight(self):
        maxWeight = 0
        for rule in self.rules:
            w = self.rules[rule].getMaxEdge()
            if w > maxWeight:
                maxWeight = w
        return maxWeight

    def getClassDict(self):
        classes = {}
        for rule in self.rules:
            if not self.rules[rule].rule.phenotype in classes:
                classes[self.rules[rule].rule.phenotype] = [self.rules[rule].myIndex]
            else:
                classes[self.rules[rule].rule.phenotype].append(self.rules[rule].myIndex)
        return classes

    def getVertices(self):
        return self.rules.keys()


class Vertex:
    def __init__(self,rule,rules,myIndex):
        self.rule = rule
        self.outgoingEdges = {}
        self.myIndex = myIndex
        for ruleIndex in range(len(rules)):
            if ruleIndex != myIndex:
                self.outgoingEdges[ruleIndex] = 0

    def addOutgoingEdgeTo(self,ruleIndex,addedWeight):
        if ruleIndex == self.myIndex:
            raise Exception("Can't add self directing edge")
        self.outgoingEdges[ruleIndex] += addedWeight

    def getMostRelatedRules(self):
        #Return reversed sorted dictionary by values
        return {k: v for k, v in reversed(sorted(self.outgoingEdges.items(), key=lambda item: item[1]))}

    def substractFromAllEdges(self,amount=1):
        for edge in self.outgoingEdges:
            self.outgoingEdges[edge] = max(0,self.outgoingEdges[edge]-amount)

    def getMaxEdge(self):
        maxEdge = 0
        for edge in self.outgoingEdges:
            if self.outgoingEdges[edge] > maxEdge:
                maxEdge = self.outgoingEdges[edge]
        return maxEdge