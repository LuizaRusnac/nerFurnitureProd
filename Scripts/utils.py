import csv

def write_text(filename, text, access_mode = 'w', map_val = None):
    with open(filename, access_mode, encoding="utf8") as f:
        for item in text:
            if map_val == 'map':
                return f.write("\t".join(map(str, item)) + "\n")
            else:
                f.write("%s\n" % item)

def write_to_text(data, filename):
    with open(filename, "w", encoding="utf-8") as text_file:
        for row in data:
            text_file.write("\t".join(map(str, row)) + "\n")
            
def read_text(filename):
    with open(filename, 'r', encoding="utf8") as f:
        text = f.readlines()
    return text

def write_csv(filename, text, acces_mode = 'w'):
    with open(filename, acces_mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ ''.join(text)])

def read_csv(filename):
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f, delimiter='\n')
        return([''.join(row) for row in csv_reader])
    