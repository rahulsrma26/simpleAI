for filepath in "$1"*_b.pgm; do
    filename=$(echo "$filepath" | cut -f 1 -d '.')
    basename=$(echo "$filename" | cut -f 1 -d '_')
    convert -size 28x84 xc:white \
        \( "$basename"_a.pgm             \) -geometry +0+0  -composite \
        \( "$basename"_b.pgm -scale 300% \) -geometry +8+30 -composite \
        \( "$basename"_c.pgm             \) -geometry +0+56 -composite \
           "$basename"_d.png
done

convert "$1"*_d.png +append -scale 200% autoencoder.png
