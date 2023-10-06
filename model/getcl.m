function [CL] = getcl(data)
    %data = data';%ecg用的别忘了改回来
    for i=1:length(data)-1

    resa(i) = abs(data(i+1,1)-data(i,1)); % 两点差的绝对值

    end

    res = sum(resa); %两点差的绝对值之和

    CL = (length(data)-1).\res; %x = B.\A 用 A 的每个元素除以 B 的对应元素

end