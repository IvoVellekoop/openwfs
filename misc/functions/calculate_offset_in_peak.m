function peak_offset = calculate_offset_in_peak(image_with_a_peak)
% calculates the offset of the peak (maximum value) from (1,1)
% Useful in finding the the shift after computing the cross-correlation between 
% two (aperiodic) images
% If (offset > end/2) then offset = -offset
%
% input:
% image_with_a_peak: an image with a single peak 
%                       (Eg: cross correlation between two (aperiodic) images)
%
% output:
% peak_offset: offset of the peak from (1,1). 
%       The directions are followed as 
%
% Beware: Only for offsets upto end/2 (in each dimensions)
%           Otherwise, large positive shift = small -ve shift 
   
[~, imax] = max(image_with_a_peak(:));
size_of_image = size(image_with_a_peak);
[peak_loc1, peak_loc2] = ind2sub(size_of_image, imax);

% the offset is calculated from (1,1). Hence, subtracting 1.
peak_offset = [peak_loc1-1 peak_loc2-1]; 

% if the offset > end/2\, it is a negative offset and is compensated accordingly 
for count_dimension = 1:length(peak_offset)
    if peak_offset(count_dimension) > size_of_image(count_dimension)/2
        peak_offset(count_dimension) = peak_offset(count_dimension) - size_of_image(count_dimension);
    end
end





