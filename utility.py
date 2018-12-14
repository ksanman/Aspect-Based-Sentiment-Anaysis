with open('../review_data/train.ft.txt', 'r') as f:
    for line in f:
        if '__label__2' in line:
            with open('../review_data/train_pos.txt', 'a+') as p:
                p.write(line)
        else:
            with open('../review_data/train_neg.txt','a+') as n:
                n.write(line) 
		
