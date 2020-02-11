%% Stoch hw
%% Problem 1
%a
clc; clear all; close all;
find_prob(1e6, 18, 'reg', 3, 6)

%b
find_prob(1e6, 18, 'fun', 3, 6)

%c
% told to skip because it is too small
% how we would do tho: (don't run it'll take too long)
% find_prob(1e13, 18, 'abs', 0,0)

% d
find_prob(1e6, 9, 'abs', 0, 0)


%% Problem 2
clc; clear all; close all;
% a-1 the average number of hit points that each troll has
find_avg(1e6, 1, 4)
% a-2 the average amount of FIREBALL damage
find_avg(1e6, 2, 2)
% a-3
find_cdf(1e6, 4, 4, 2, 2)

% b indexing array is pmf function
pmf_fireball = find_pmf(1e6, 2, 4, 2, 2)
pmf_hptroll = find_pmf(1e6, 1, 4, 1, 4)

% c
% find_cdf(1e6, 1, 2, 1, 4)^6 * find_prob(1e6, 2, 'reg', 2, 2) + find_cdf(1e6, 1, 3, 1, 4)^6 * find_prob(1e6, 3, 'reg', 2, 2) + find_cdf(1e6, 1, 4, 1, 4)^6 * find_prob(1e6, 4, 'reg', 2, 2)
kill_all_six_trolls(1e6)
% d
kill_all_but_one_six_trolls(1e6)

% e
find_avg_shitvam_hit(1e6)

%%
% f
% 0% dont make me laugh lol xoxo

%%
% functions
function avg_shitvam_hit = find_avg_shitvam_hit(num_trials)
    avg_shitvam_hit = 0;
    for i = 1:num_trials
        if roll_dice(1, 20) >= 11
            avg_shitvam_hit = avg_shitvam_hit + roll_dice(2, 6);
            if roll_dice(1, 20) >= 11
                avg_shitvam_hit = avg_shitvam_hit + roll_dice(1, 4);
            end
        end
    end
    avg_shitvam_hit = avg_shitvam_hit/num_trials;
end


function avg_hp = kill_all_but_one_six_trolls(num_trials)
    hp = 0;
    cnt = 0;
    for i = 1:num_trials
        troll_hps = randi(4,1,6);
        fireball = roll_dice(2, 2);
        troll_hps = troll_hps - fireball;
        c = positive_unique(troll_hps);
        if c > 0
            hp = hp + c;
            cnt = cnt + 1;
        end
    end 
    cnt
    avg_hp = hp / cnt;
end

function c = positive_unique(hps)
    cnt = 0;
    c = 0;
    for i = 1:length(hps)
        if hps(i) > 0
            c= hps(i);
            cnt = cnt + 1;
            if (cnt > 1)
                c=0;
                return
            end
        end
    end
end

function prob = kill_all_six_trolls(num_trials)
    succ = 0;
    for i = 1:num_trials
        troll_hps = randi(4,1,6);
        fireball = roll_dice(2, 2);
        if fireball >= max(troll_hps)
            succ = succ + 1;
        end
    end 
    prob = succ / num_trials;
end

function pmf = find_pmf(num_trials, begin, last, num_rolls, num_sides)
    succ_count=0;
    pmf = [];
    for num_to_find = begin:last
        for i = 1:num_trials
            if num_to_find == roll_dice(num_rolls, num_sides)
                succ_count = succ_count + 1;
            end
        end
        pmf = [pmf succ_count / num_trials];
        succ_count = 0;
    end
end

function cdf = find_cdf(num_trials, begin, last, num_rolls, num_sides)
    cdf = 0;
    succ_count=0;
    for num_to_find = begin:last
        for i = 1:num_trials
            if num_to_find == roll_dice(num_rolls, num_sides)
                succ_count = succ_count + 1;
            end
        end
        cdf = cdf + succ_count / num_trials;
        succ_count = 0;
    end
end

function avg = find_avg(num_trials, num_rolls, num_sides)
    avg = 0;
    for i = 1:num_trials
        avg = avg + roll_dice(num_rolls, num_sides);
    end
    avg = avg / num_trials;
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
    elseif strcmp(method, 'abs')
        for i = 1:num_trials
           succ_count = succ_count + ab_scores(6, num_to_find);
        end
    end
 
    prob = succ_count / num_trials;
    
    
end

function succ = ab_scores(num_scores, num_to_get)
    succ = 0;
    for i = 1:num_scores
        if num_to_get ~= fun_method(3,6)
            return
        end
    end
    succ = 1;
end

function result = fun_method(num_times, num_sides)
    result = 0;
    for i = 1:3
        one_roll = roll_dice(num_times,num_sides);
        result = max(result, one_roll);
    end
end

function result = roll_dice(num_times,num_sides)
    result = sum(randi(num_sides,1,num_times));
end
