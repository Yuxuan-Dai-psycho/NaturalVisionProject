function responseEvaluation(target,response,condName)
% target, response,rt,condName

idx = any(response,2);% only keep trial with response
[cVal,cMat,~,cPer] = confusion(target(idx,:)',response(idx,:)');
figure('Units','normalized','Position',[0 0 1 1])
subplot(1,2,1), imagesc(cMat);
title(sprintf('KeyProp = %.2f, Accuracy = %.2f',sum(idx)/length(target) ,1-cVal));
axis square
set(gca,'Xtick',1:length(cMat), 'XTickLabel',condName,...
    'Ytick',1:length(cMat),'YTickLabel', condName);
colorbar
subplot(1,2,2), bar(cPer);
set(gca,'XTickLabel', condName);
ylabel('Rate')
axis square
legend({'Miss','False alarm','Hit','Correct reject'},...
   'Orientation','vertical' ,'Location','northeastoutside' )
legend boxoff
