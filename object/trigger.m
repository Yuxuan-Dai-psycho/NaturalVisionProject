%%% 代码前放, 获取地址,0378是那边opm主机的
ioObj = io64;
status = io64(ioObj);
address = hex2dec('0378');

%%%  在刺激演示阶段中发trigger 
%%%  这是我们的可以参考如何加进去，注意只能发送四个event_id ，1234标识四个事件，所以要
%%%  选一下，但是如果你们考察的event四个id足够标识了，那就没问题了
%%%  比如以下表示 1 代表 stimlus；2 标识iti， 3代表cue呈现阶段， 每个标记后发送一个0

%trigger for stimulus
 timere = GetSecs;
 io64(ioObj,address,1);
 while GetSecs-timere<0.01
 end
 io64(ioObj,address,0);
  
  
 %%%  trigger for ITI:
 timere = GetSecs;
 io64(ioObj,address,2);
 while GetSecs-timere<0.01
 end
 io64(ioObj,address,0);
  
 %%%  trigger for cue:
 timere = GetSecs;
 io64(ioObj,address,3);
 while GetSecs-timere<0.01
 end
 io64(ioObj,address,0);
  