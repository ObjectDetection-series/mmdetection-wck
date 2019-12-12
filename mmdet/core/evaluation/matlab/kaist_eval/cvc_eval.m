% Author:Yuan Yuan
% Date:2019/02/13
% Description: This file is modified based on project of 'rgbt-ped-detection'
%                     Used for evaluating average miss rate in pedestrian
%                     detection.This script depends on 'bbGt2.m' and
%                     Piotr's Computer Vision Matlab Toolbox,Version 3.26.
fprintf('\ncvc dataset evaluating...\n');
%% parameters setting
pLoad={'lbls',{'person'},'ilbls',{'people','person?','cyclist'},'squarify',{3,.41}};  % for traing and test (common)
pLoad = [pLoad, 'hRng',[50 inf], 'vType',{{'none'}},'xRng',[5 635],'yRng',[5 475]];  % for testing config
reapply = 1;
thr = 0.5;
mul = 0;
show = 0;
lims = [3.1e-3 1e1 .05 1];
ref = 10.^(-2:.25:0);
dataDir = ['/media/',getenv('USER'),'/Data/DoubleCircle/datasets/CVC'];
gtDir = [dataDir,'/annotations'];
dtDir = [dataDir,'/res'];
subset =[dataDir,'/imageSets/test-all.txt'];

%% evaluating detection results
[gt,dt]=bbGt2('loadAll',gtDir,dtDir,pLoad,subset);
max_score_list = [];
for ii=1:length(gt)
    if ~isempty(gt{ii})
        gt{ii}(:,1) = gt{ii}(:,1) - gt{ii}(:,3)/2;
        gt{ii}(:,2) = gt{ii}(:,2) - gt{ii}(:,4)/2;
    end
end
for ii=1:length(dt)
    if ~isempty(dt{ii})
        dt{ii}(:,3) = dt{ii}(:,3) - dt{ii}(:,1) + 1;
        dt{ii}(:,4) = dt{ii}(:,4) - dt{ii}(:,2) + 1;
        max_score_list = [max_score_list,max(dt{ii}(:,5))];
    end
end
max_score = max(max_score_list);
[gt,dt] = bbGt2('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt2('compRoc',gt,dt,1,ref);
miss=exp(mean(log(max(1e-10,1-miss))));
roc=[score fp tp];
fprintf('\nlog-average miss rate = %.2f%%\n',miss*100);

fid = fopen('/media/ser606/Data/DoubleCircle/temp/temp.txt','r');
str = fgets(fid);
fclose(fid);

str = [str,'/eval_result.txt'];
fid = fopen(str,'a+');
fprintf(fid,'log-average miss rate = %.2f%%\n',miss*100);
fclose(fid);

% optionally plot roc
if show > 0
    figure(show);
    plotRoc([fp tp],'logx',1,'logy',1,'xLbl','False positives per image',...
      'lims',lims,'color',clr,'smooth',1,'fpTarget',ref);
    title(sprintf('log-average miss rate = %.2f%%',miss*100));
end
% savefig([name 'Roc'],show,'png');
% print(gcf, '-dpng', [name '-' figName '-Roc.png'], '-r300');      % Use built-in function