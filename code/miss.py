# from pipeline.parse_source()
import pandas as pd
import os

label_file = ["camel-1.4.csv"]
src_dir = ["apache-camel-1.4.0"]
dpath = "dataset/"

data = pd.read_csv(dpath + src_dir[0] + '/' + label_file[0])
data['id'] = range(data.shape[0])
data = data.rename(columns = {'name.1':'code', 'bug':'label'})
data['path'] = data['code']
def parse_program(loc):
    javafile = dpath+src_dir[0]+'/src/main/java/'+loc.replace('.','/')+'.java'
    scalafile = dpath+src_dir[0]+'/src/main/java/'+loc.replace('.','/')+'.scala'
    if os.path.exists(scalafile):
        os.rename(scalafile,javafile)
    #print (javafile)
    if os.path.exists(javafile):
        fd = open(javafile, "r")
        func = fd.read()
        import javalang
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = None
        try:
            tree = parser.parse_compilation_unit()
        except:
            print ('!!!',javafile)
        return tree
    else:
        return None
data['code'] = data['code'].apply(parse_program)
data = data[data['code'].isnull()!=False]
print (data[['id','path','code','label']])
