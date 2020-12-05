function binResponseEvaluation(target,response,condName)
% binResponseEvaluation(target,response,condName)
% target, response,rt,condName

idx = any(response,2);% only keep trial with response
[cVal,cMat,~,cPer] = binConfusion(target(idx,:)',response(idx,:)');
figure('Units','normalized','Position',[0 0 1 1])
% subplot(1,2,1), 
imagesc(cMat);
title(sprintf('RespProp = %.2f, Accuracy = %.2f',sum(idx)/length(target) ,1-cVal));
axis square
set(gca,'Xtick',1:length(cMat), 'XTickLabel',condName,...
    'Ytick',1:length(cMat),'YTickLabel',condName);
colorbar
text(0.75,1,num2str(round(cPer(1,3)*1000)/10),'FontSize',50,'Color','r');% hit
text(0.75,2,num2str(round(cPer(1,1)*1000)/10),'FontSize',50,'Color','r');% miss
text(1.75,1,num2str(round(cPer(1,2)*1000)/10),'FontSize',50,'Color','r');% false alarm
text(1.75,2,num2str(round(cPer(1,4)*1000)/10),'FontSize',50,'Color','r');% corect reject

% subplot(1,2,2), bar(cPer);
% set(gca,'XTickLabel',condName);
% ylabel('Rate')
% axis square
% legend({'Miss','False alarm','Hit','Correct reject'},...
%    'Orientation','vertical' ,'Location','northeastoutside' )
% legend boxoff
