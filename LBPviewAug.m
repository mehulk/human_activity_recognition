
maxActivity = 20;
maxSubject = 10;
maxTask = 3;
path_name_structure = '%s/train/%d';

%front
X_train_front=[];
y_train_front=[];

%Side
X_train_side=[];
y_train_side=[];

%Top
X_train_top=[];
y_train_top=[];
i=0;

for an = 1:maxActivity
    path_name=sprintf(path_name_structure,'F',an);      
    fil = fullfile(path_name,'*.jpg');
    d=dir(fil);
    for k=1:numel(d)
      filename=fullfile(path_name,d(k).name);
      [LBPf] = LBPfeatures(filename);      
      X_train_front=[X_train_front;LBPf]; %#ok<*AGROW>
      y_train_front=[y_train_front;an];
      disp(filename);
      i=i+1;
      disp(i);
    end
    
end

disp('Front Done');

for an = 1:maxActivity
    path_name=sprintf(path_name_structure,'S',an);      
    fil = fullfile(path_name,'*.jpg');
    d=dir(fil);
    for k=1:numel(d)
      filename=fullfile(path_name,d(k).name);
      [LBPf] = LBPfeatures(filename);    
      X_train_side=[X_train_side;LBPf];
      y_train_side=[y_train_side;an];
    end
    
end

disp('Side Done');

for an = 1:maxActivity
    path_name=sprintf(path_name_structure,'T',an);      
    fil = fullfile(path_name,'*.jpg');
    d=dir(fil);
    for k=1:numel(d)
      filename=fullfile(path_name,d(k).name);
      [LBPf] = LBPfeatures(filename);  
      X_train_top=[X_train_top;LBPf];
      y_train_top=[y_train_top;an];
    end
    
end

X_train = [X_train_front X_train_side X_train_top];
y_train = y_train_front;

save('X_train','X_train');
save('y_train','y_train');
