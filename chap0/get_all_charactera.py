import string

for item in [string.ascii_letters, string.digits, string.punctuation]:
    print("{}\t{}".format(len(item), item))
