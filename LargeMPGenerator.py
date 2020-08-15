import random
import csv

# Generate group and instance labeled MP problems

class LargeMPGenerator:
    def __init__(self,bits,num_instances,filepath):
        # Check if bits is valid from 6 bit to 135 bit MP
        is_valid = False
        address_bit_count = 0
        register_bit_count = 0
        for i in range(2, 8):
            if bits == i + pow(2, i):
                is_valid = True
                address_bit_count = i
                register_bit_count = bits - address_bit_count

        if not is_valid:
            raise Exception("# of bits is invalid. Must be n+2^n, 2 <= n <= 7")

        instances = []
        instance_id = 0
        for i in range(num_instances):
            s = ''
            for bit in range(bits):
                s += str(random.randint(0,1))

            address = s[:address_bit_count]
            registers = s[-register_bit_count:]

            group_id = self.binaryToDecimal(int(address))
            phenotype = registers[group_id]

            row = []
            row.append(instance_id)
            row.append(group_id)
            for a in address:
                row.append(int(a))
            for r in registers:
                row.append(int(r))
            row.append(phenotype)

            instances.append(row)
            instance_id += 1

        headers = ['Instance','Group']
        for i in range(address_bit_count):
            headers.append('A'+str(i))
        for i in range(register_bit_count):
            headers.append('R'+str(i))
        headers.append('Class')

        with open(filepath,mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            for row in instances:
                writer.writerow(row)
        file.close()

    def binaryToDecimal(self,binary):
        decimal, i, n = 0, 0, 0
        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1
        return decimal



if __name__ == '__main__':
    LargeMPGenerator(bits=37,num_instances=5000,filepath='Datasets/mp37_5k.csv')
    LargeMPGenerator(bits=70,num_instances=10000,filepath='Datasets/mp70_10k.csv')
    LargeMPGenerator(bits=135,num_instances=30000,filepath='Datasets/mp135_30k.csv')
