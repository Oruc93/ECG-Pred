function [words, symbols] = calc_symboldynamics(beat_to_beat_intervals, a, mode)

%a = 0.05;
mu = mean(beat_to_beat_intervals);
symbols = zeros(length(beat_to_beat_intervals), 1);
if mode == "mean"
    for k=1:length(beat_to_beat_intervals)
       if (mu < beat_to_beat_intervals(k)) && (beat_to_beat_intervals(k) <= (1+a)*mu)
           symbols(k) = 0;
       elseif (1+a)*mu < beat_to_beat_intervals(k)
           symbols(k) = 1;
       elseif (1-a)*mu < beat_to_beat_intervals(k) && beat_to_beat_intervals(k) <= mu
           symbols(k) = 2;
       else
           symbols(k) = 3;
       end
    end
elseif mode == "diff"
    diff = zeros(length(beat_to_beat_intervals));
    for k=2:length(beat_to_beat_intervals)
        diff(k-1) = beat_to_beat_intervals(k) - beat_to_beat_intervals(k-1);
    end
    sigma_delta = std(diff);
    for k = 1:length(diff)
       if (diff(k) > a * sigma_delta)
            symbols(k) = 1;
       elseif (diff(k) > 0) && (diff(k) <= a * sigma_delta)
           symbols(k) = 0;
       elseif (diff(k) > (-1*a*sigma_delta)) && (diff(k) > 0)
           symbols(k) = 2;
       elseif (diff(k) < -(a*sigma_delta))
           symbols(k) = 3;
       end
    end
elseif mode == "movdiff"
    diff = zeros(length(beat_to_beat_intervals) , 1);
    for k=2:length(beat_to_beat_intervals)
        diff(k-1) = beat_to_beat_intervals(k) - beat_to_beat_intervals(k-1);
    end
    for k = 1:length(diff)-1
        if diff(k) > 0            
            %if diff(k) <= beat_to_beat_intervals(k+1) * a          % Änderung zu Zeile drunter
            if diff(k) <= beat_to_beat_intervals(k) * a    
                symbols(k) = 0;
            else
                symbols(k) = 1;
            end
        else
            %if abs(diff(k)) <= beat_to_beat_intervals(k+1) * a     % Änderung zu Zeile drunter
            if abs(diff(k)) <= beat_to_beat_intervals(k) * a    
                symbols(k) = 2;
            else
                symbols(k) = 3;
            end
        end
    end    
    %symbols = symbols(2:end-2);
   
end
     words = zeros(64, 1);
    for t=2:(length(symbols)-4) %Potenzieller off by one [Bug 4 bug vs 2.0]
        arrpos = symbols(t) * 16 + symbols(t+1) * 4 + symbols(t+2) * 1;
        words(arrpos + 1) = words(arrpos + 1) + 1;
    end
end


