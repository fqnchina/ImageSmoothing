clear all;
addpath('./util');

inputDir = '.\test_detail_enhance\';
outputDir = '.\test_detail_enhance\';
images = dir([inputDir '*-input.png']);
for m = 1:length(images)
    inputname = [inputDir images(m).name];
    outputname = strrep(inputname,'input','detail-enhance');
    input = double(imread(inputname))/255;
    [rows, columns, numberOfColorChannels] = size(input);
    if numberOfColorChannels == 1
        temp = input;
        input = zeros(rows,columns,3);
        input(:,:,1) = temp;
        input(:,:,2) = temp;
        input(:,:,3) = temp;
    end
    cform = makecform('srgb2lab');
    input_lab = applycform(input, cform);
    input_l = input_lab(:,:,1);
    
    smoothname = strrep(inputname,'input.png','smooth.png');
    smooth = double(imread(smoothname))/255;
    smooth_lab = applycform(smooth, cform);
    smooth_l = smooth_lab(:,:,1);

    val0 = 15;
    val2 = 1;
    exposure = 1.0;
    saturation = 1.0;
    gamma = 1.0;

    output = tonemapLAB_simple(input_lab,smooth_l,input_l,val0,val2,exposure,gamma,saturation);

    imwrite(output, outputname);
end