function XxWriteTiffLargerThan4GB(data, file_out, dtype)

if nargin < 3, dtype = 16; end

tif_tag.ImageLength = size(data,1); %y dim
tif_tag.ImageWidth = size(data,2);  %x dim
tif_tag.Photometric = Tiff.Photometric.MinIsBlack;
tif_tag.BitsPerSample = dtype;
tif_tag.SamplesPerPixel = 1;
tif_tag.SampleFormat = Tiff.SampleFormat.UInt;
tif_tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%tif_info_struct.Compression=Tiff.Compression.PackBits;
tif_tag.Compression = Tiff.Compression.None;


t = Tiff(file_out,'w8');
t.setTag(tif_tag);
t.write(data(:,:,1));
t.close;
for i = 2:size(data,3)
    t = Tiff(file_out,'a');
    t.setTag(tif_tag);
    t.write(data(:,:,i));
    t.close;
end
