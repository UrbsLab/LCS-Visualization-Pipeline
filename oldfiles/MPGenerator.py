
import random
import copy
import csv

class MPGenerator:
    def __init__(self,bits=6,classCount=2,do_instances=True,do_groups=True):
        #Check if bits is valid from 6 bit to 135 bit MP
        isValid = False
        addressBitCount = 0
        registerBitCount = 0
        for i in range(2,8):
            if bits == i+pow(2,i):
                isValid = True
                addressBitCount = i
                registerBitCount = bits - addressBitCount

        if not isValid:
            raise Exception("# of bits is invalid. Must be n+2^n, 2 <= n <= 7")

        #Check if classCount is valid
        if classCount < 2:
            raise Exception("# of classes must be at least 2")

        #Populate list of possible MP conditions and actions
        mps = []
        instanceCount = 0
        groupCount = 0
        for i in range(pow(2,addressBitCount)):
            address = str(self.baseTenToBase(i,2))
            for j in range(pow(classCount,registerBitCount)):
                register = str(self.baseTenToBase(j,classCount))
                while len(register) < registerBitCount:
                    register = "0"+register
                while len(address) < addressBitCount:
                    address = "0"+address
                nAddress = []
                nRegister = []
                for z in address:
                    nAddress.append(int(z))
                for z in register:
                    nRegister.append(int(z))
                r = self.binaryToDecimal(int(address))
                phenotype = register[r]
                if not do_instances and not do_groups:
                    mps.append(nAddress + nRegister + [int(phenotype)])
                if do_instances and not do_groups:
                    mps.append(nAddress + nRegister + [int(phenotype)]+['filler'])
                if not do_instances and do_groups:
                    mps.append(nAddress + nRegister + [int(phenotype)]+[int(groupCount)])
                if do_instances and do_groups:
                    mps.append(nAddress + nRegister + [int(phenotype)]+['filler']+[int(groupCount)])
                instanceCount += 1
            groupCount += 1

        self.instances = mps
        self.bits = bits
        self.classCount = classCount
        self.do_instances = do_instances
        self.do_groups = do_groups
        self.addressbitcount = addressBitCount

    #Returns num in base endBase
    def baseTenToBase(self,num, endBase):
        s = ""
        while int(num / endBase) != 0:
            r = num % endBase
            num = int(num / endBase)
            s = str(r) + s
        r = num % endBase
        s = str(r) + s
        return int(s)

    def binaryToDecimal(self,binary):
        binary1 = binary
        decimal, i, n = 0, 0, 0
        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1
        return decimal

    def exportInstances(self,filename='generated.csv',instanceCount=None):
        if instanceCount == None:
            instanceCount = len(self.instances)

        filename = 'Datasets/'+filename+'.csv'

        headerNames = []
        for i in range(self.bits+1):
            if i != self.bits:
                if i < self.addressbitcount:
                    headerNames.append("A"+str(i))
                else:
                    headerNames.append("R" + str(i))
            else:
                if self.do_instances and self.do_groups:
                    headerNames.append("Class")
                    headerNames.append("Instance")
                    headerNames.append("Group")
                elif self.do_instances and not self.do_groups:
                    headerNames.append("Class")
                    headerNames.append("Instance")
                elif not self.do_instances and self.do_groups:
                    headerNames.append("Class")
                    headerNames.append("Group")
                else:
                    headerNames.append("Class")
        with open(filename,mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headerNames)
            if instanceCount <= len(self.instances):
                samples = random.sample(self.instances,k=instanceCount)
            else:
                samples = random.sample(self.instances, k=len(self.instances))
                added = random.choices(self.instances,k=instanceCount-len(self.instances))
                samples += added
            iCount = 0
            for i in range(len(samples)):
                samples[i] = copy.deepcopy(samples[i])
                for j in range(len(samples[i])):
                    if samples[i][j] == 'filler':
                        samples[i][j] = iCount
                iCount += 1

            for inst in samples:
                writer.writerow(inst)

if __name__ == '__main__':
    m = MPGenerator(bits=6,classCount=2,do_instances=False)
    m.exportInstances('mp6_noinstances',instanceCount=500)

    m = MPGenerator(bits=6, classCount=2,do_groups=False)
    m.exportInstances('mp6_nogroups', instanceCount=500)

    m = MPGenerator(bits=6, classCount=2,do_instances=False,do_groups=False)
    m.exportInstances('mp6_none', instanceCount=500)

    m = MPGenerator(bits=6, classCount=2)
    m.exportInstances('mp6_full', instanceCount=500)

    m = MPGenerator(bits=11, classCount=2)
    m.exportInstances('mp11_full', instanceCount=2000)

    m = MPGenerator(bits=20, classCount=2)
    m.exportInstances('mp20_full', instanceCount=5000)