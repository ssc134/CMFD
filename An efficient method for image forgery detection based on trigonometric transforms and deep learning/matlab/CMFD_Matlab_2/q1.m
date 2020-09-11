clc;
% UxV image dimension
%U = 256;
%V = 256;
U = 532;
V = 800;
% BxB filter/block dimension
%B = 10;
B = 100;
% match_feat threshold
Threshold = 5;
% black map
black_map = uint8(zeros(U, V));
% new image
img2 = uint8(zeros(U, V));

%rgb_img = imread("img.jpg");
rgb_img = imread("tamp_1.jpg");
gsc_img = rgb2gray(rgb_img);
img1 = imresize(gsc_img, [U V]);
%imshow(img1);
%whos;

mb = ones((U - B + 1) * (U - B + 1), 5);
row_count = 1;

for i = 1:1:U - B + 1

    for j = 1:1:V - B + 1
        % choosing a B sized block and convolving
        temp = img1(i:i + B - 1, j:j + B - 1);
        temp = dct2(temp);

        % si stands for sum of elements in quadrant/inner circle i
        s1 = 0;
        s2 = 0;
        s3 = 0;
        s4 = 0;

        % 1st inner circle sum
        for k = 1:1:floor(B / 2)

            for l = 1:1:floor(B / 2)
                s1 = s1 + temp(k, l);
            end

        end

        % 2nd quadrant circle sum
        for k = 1:1:floor(B / 2)

            for l = floor(B / 2) + 1:1:B
                s2 = s2 + temp(k, l);
            end

        end

        % 3rd quadrant circle sum
        for k = floor(B / 2) + 1:1:B

            for l = 1:1:floor(B / 2)
                s3 = s3 + temp(k, l);
            end

        end

        % 4th quadrant circle sum
        for k = floor(B / 2) + 1:1:B

            for l = floor(B / 2) + 1:1:B
                s4 = s4 + temp(k, l);
            end

        end

        mb(row_count, 1) = s1;
        mb(row_count, 2) = s2;
        mb(row_count, 3) = s3;
        mb(row_count, 4) = s4;
        mb(row_count, 5) = row_count;
        row_count = row_count + 1;
    end

end

% correct row_count value from the previous for loop
row_count = row_count - 1;

% lexicographical sorting of vectors
sortrows(mb);

% matching match_feat among adjacent blocks in mb and
% making a note in black_map if match_feat>=Threshold
for i = 2:1:row_count
    match_feat = int32(0);

    for j = 1:1:4
        match_feat = match_feat + (mb(i, j) - mb(i - 1, j))^2;
    end

    if match_feat <= Threshold

        quotient = int16((i - 1) / (V - B + 1));
        remiander = int16(mod((i - 1), V - B + 1));
        block1_row = 0;
        block1_column = 0;

        if remiander == 0
            block1_row = quotient;
            block1_column = V - B + 1;
        else
            block1_row = quotient + 1;
            block1_column = remiander;
        end

        quotient = int16(i / (V - B + 1));
        remiander = int16(mod(i, V - B + 1));
        block2_row = 0;
        block2_column = 0;

        if remiander == 0
            block2_row = quotient;
            block2_column = V - B + 1;
        else
            block2_row = quotient + 1;
            block2_column = remiander;
        end

        for k = block1_row:1:block1_row + B - 1

            for l = block1_column:1:block1_column + B - 1
                black_map(k, l) = 255;
            end

        end

        for k = block2_row:1:block2_row + B - 1

            for l = block2_column:1:block2_column + B - 1
                black_map(k, l) = 255;
            end

        end

    end

end

for i = 1:1:U

    for j = 1:1:V

        if black_map(i, j) == 255
            img2(i, j) = 0;
        else
            img2(i, j) = img1(i, j);
        end

    end

end

%{
img3 = uint8(zeros(U,V));
for i = 1:1:U
    for j = 1:1:V 
        img3(i,j) = img1(i,j);
    end
end
%}

%imshow(img2);
%imshow(black_map);

subplot(1,3,1);
imshow(img1)%, "Original Grayscale");
subplot(1,3,2);
imshow(img2)%, "Forgery Detected Grayscale");
subplot(1,3,3);
imshow(black_map)%, "Black_Map");


imwrite(img2, "Forgery_Detected Grayscale.jpg");
imwrite(black_map, "Black_Map.jpg");

disp("Done");