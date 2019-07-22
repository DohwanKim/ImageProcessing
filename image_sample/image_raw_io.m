fid = fopen('House.raw', 'rb');
I = fread(fid, [256, 256], 'uchar');
fclose(fid);

I = I';

imshow(I, [0, 256]);

fid = fopen('Houseclone.raw', 'wb');
fwrite(fid, I, 'uchar');
fclose(fid);

imwrite(uint8(I),'House.bmp', 'bmp') 