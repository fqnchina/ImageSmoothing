clear all;
addpath('./util');

inputDir = './test_stylization/';
outputDir = './test_stylization/';
images = dir([inputDir '*-smooth.png']);
for m = 1:length(images)
    inputname = [inputDir images(m).name];
    strokename = [outputDir images(m).name];
    strokename = strrep(strokename,'smooth.png','stroke.png');
    stylename = [outputDir images(m).name];
    stylename = strrep(stylename,'smooth.png','draw.png');

    input = im2double(imread(inputname));
    output_style = PencilDrawing(input, 8, 1, 8, 2, 1, 0);
    imwrite(output_style,stylename);
    
    input_ycbcr = rgb2ycbcr(input);
    input_y = input_ycbcr(:,:,1);
    output_stroke = GenStroke(input_y, 8, 1, 8).^2;
    imwrite(output_stroke,strokename);
end