unoconv --listener &
for file in *txt
    do 
        unoconv -f pdf "$file"   
    done
