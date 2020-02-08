%% Stoch hw
% comment a lot cuz he likes that

%% Problem 1
%a
find_prob(1e6, 18, 'reg', 3,6)

%b
find_prob(1e6, 18, 'fun', 3,6)

%c
%told to skip because it is too small
% how we would do tho: (don't run it'll take too long)
% find_prob(1e13, 18, 'abs_scores', 0,0)

% d
find_prob(1e6, 9, 'abs_scores', 0,0)


%% Problem 2

function succ = ab_scores(num_scores, num_to_get)
    succ = 0;
    for i = 1:num_scores
        if num_to_get ~= fun_method(3,6)
            return
        end
    end
    succ = 1;
end

function prob = find_prob(num_trials, num_to_find, method, num_rolls, num_sides)
    succ_count=0;
    if strcmp(method,'fun')
        for i = 1:num_trials
            if num_to_find == fun_method(num_rolls, num_sides)
                succ_count = succ_count + 1;
            end
        end
    elseif strcmp(method,'reg')
        for i = 1:num_trials
            if num_to_find == roll_dice(num_rolls, num_sides)
                succ_count = succ_count + 1;
            end
        end
    else
        for i = 1:num_trials
           succ_count = succ_count + ab_scores(6, num_to_find);
        end
    end
 
    prob = succ_count / num_trials;
    
    
end

function result = fun_method(num_times, num_sides)
    result = 0;
    for i = 1:3
        one_roll = roll_dice(num_times,num_sides);
        if one_roll > result
            result = one_roll;
        end
    end
end

function result = roll_dice(num_times,num_sides)
    result = sum(randi(num_sides,1,num_times));
end
