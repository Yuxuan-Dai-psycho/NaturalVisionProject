function Instruction(wptr,insTexture)
% Define instruction images for start and end
% Press space to continue

Screen('DrawTexture', wptr, insTexture);
Screen('Flip',wptr);

key_Space=KbName('Space');
while 1
    [~, key_Code, ~]=KbWait([], 3);     
    if key_Code(key_Space)
        break;
    end
end

Screen('FillRect',wptr,[128 128 128]);  
Screen('Flip',wptr);

end

