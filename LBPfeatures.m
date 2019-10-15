function [LBPf] = LBPfeatures(file_name)
     
     lbp = imread(file_name);
     
     if size(lbp,3)==3
         lbp = rgb2gray(lbp);
     end
     
     LBPf = extractLBPFeatures(lbp,'NumNeighbors',8);     
         
end

