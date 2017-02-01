
do  
    local data = torch.class('data_fish')

    function data:__init(args)
        self.file_path=args.file_path;
        self.batch_size=args.batch_size;
        self.limit=args.limit;
        self.augmentation=args.augmentation;
        self.bgr=args.bgr;
        self.angles={-10,-5,0,5,10}
        print ('self.augmentation',self.augmentation);

        self.start_idx=1;
        self.input_size=args.input_size;
        self.crop_size=args.crop_size;
        if not self.crop_size then
            self.crop_size=args.input_size;
        end
        self.labels_shape=1;

        self.training_set={};
        
        self.lines_data=self:readDataFile(self.file_path);
        
        if self.augmentation then
            self.lines_data =self:shuffleLines(self.lines_data);
        end

        if self.limit~=nil then
            local lines_data=self.lines_data;
            self.lines_data={};

            for i=1,self.limit do
                self.lines_data[#self.lines_data+1]=lines_data[i];
            end
        end
        print (#self.lines_data);
    end


    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;

        local shuffle = torch.randperm(len)
        
        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
        end
        return lines2;
    end

    function data:getTrainingData()
        local start_idx_before = self.start_idx

        self.training_set.data=torch.zeros(self.batch_size,3,self.crop_size[1]
            ,self.crop_size[2]);
        self.training_set.label=torch.zeros(self.batch_size,self.labels_shape);
        -- ,self.labels_shape[2]);
        self.training_set.input={};
        
        self.start_idx=self:addTrainingData(self.training_set,self.batch_size,
            self.lines_data,self.start_idx)    
        
        

        if self.start_idx<start_idx_before and self.augmentation then
            print ('shuffling data'..self.start_idx..' '..start_idx_before )
            self.lines_data=self:shuffleLines(self.lines_data);
        end

    end

    
    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            local start_idx, end_idx = string.find(line, ' ');
            local img_path=string.sub(line,1,start_idx-1);
            local img_label=string.sub(line,end_idx+1,#line);
            file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines

    end


    function data:rotateIm(img_fish,angles)
        
        local rand=math.random(#angles);
        local angle=math.rad(angles[rand]);
        img_fish=image.rotate(img_fish,angle,"bilinear");

        return img_fish
    end

    function data:cropIm(img_fish,crop_size)
        local img_size=img_fish:size();
        assert (crop_size[1]==crop_size[2]);
        assert (img_size[2]==img_size[3]);
        assert (img_size[2]>=crop_size[1]);
        local diff=img_size[2]-crop_size[1];
        local start_x=math.random(diff)-1;
        local start_y=math.random(diff)-1;
        img_fish=image.crop(img_fish,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        assert (img_fish:size(2)==crop_size[1] and img_fish:size(3)==crop_size[2])
        return img_fish;
    end

    function data:processIm(img_fish)
        
        img_fish:mul(255);
        
        -- if img_fish:size(2)~=self.input_size[1] then 
        --     img_fish = image.scale(img_fish,self.input_size[1],self.input_size[2]);
        -- end
        
        
        if self.augmentation then
            -- crop, then flip or rotate
            img_fish=self:cropIm(img_fish,self.crop_size);
            local rand=math.random(2);
            if rand==1 then
                image.hflip(img_fish,img_fish);
            end
            img_fish=self:rotateIm(img_fish,self.angles);
        else
            img_fish = image.scale(img_fish,self.crop_size[1],self.crop_size[2]); 
        end

        return img_fish
    end

    function data:addTrainingData(training_set,batch_size,lines_data,start_idx)
        local list_idx=start_idx;
        local list_size=#lines_data;
        local curr_idx=1;
        while curr_idx<= batch_size do
            local img_path_fish=lines_data[list_idx][1];
            local label_path_fish=lines_data[list_idx][2];
            
            local status_img_fish,img_fish=pcall(image.load,img_path_fish);
            
            if status_img_fish then
                local label_fish=tonumber(label_path_fish)+1;

                if img_fish:size()[1]==1 then
                    img_fish= torch.cat(img_fish,img_fish,1):cat(img_fish,1)
                end
                -- img_fish,label_fish=self:processImAndLabel(img_fish,label_fish,params)
                img_fish=self:processIm(img_fish);
                if self.bgr then
                    -- print 'bgring';
                    local img_fish_temp=img_fish:clone();
                    img_fish[{1,{},{}}]=img_fish_temp[{3,{},{}}];
                    img_fish[{3,{},{}}]=img_fish_temp[{1,{},{}}];
                end
                
                training_set.label[curr_idx]=label_fish;
                training_set.input[curr_idx]=img_path_fish;
                training_set.data[curr_idx]=img_fish;
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        return list_idx;
    end

    
end

return data