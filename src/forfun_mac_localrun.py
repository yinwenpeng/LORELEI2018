




def generate_filelist():
    writefile = open('/Users/yinwenpeng/Downloads/LORELEI/wp_filelist.txt', 'w')
    readfile = open('/Users/yinwenpeng/Downloads/LORELEI/annotated_filelist_SF.tab', 'r')
    co = 0
    written_set= set()
    for line in readfile:
        if co>0:
            parts = line.strip().split('\t')
            doc_id = parts[0]
            if doc_id not in written_set:
                written_set.add(doc_id)
                writefile.write(parts[0]+'\tTRUE\n')
        co+=1
    readfile.close()
    writefile.close()

if __name__ == '__main__':
    generate_filelist()
