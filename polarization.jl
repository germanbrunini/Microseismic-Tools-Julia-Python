function polarization(tx::Array{Float64,1},
                      ty::Array{Float64,1},
                      tz::Array{Float64,1},
                      ns_::Int64,
                      nt_::Int64,
                      dt_::Float64)

    #### CAREFULLY
    #### assign half-length of the moving time window
    halfInt    = 0.5*0.005;          # in seconds
    sampledint = halfInt/dt_;       # how many samples in that interval
    w = Int64(round(sampledint));   # integer of sampledinterval

    trstart = Int64(w + 1);         # start and ... end time for the analysis
    trend   = Int64(ns_ - w);       # ... end time for the analysis.

    begin1    = copy(trstart);       # start at sample w+1.
    end1      = copy(trend);         # finish at sample ns-w.
    alphasize = end1 - begin1;


    alpha1   = zeros(alphasize,nt_); # rectilinearity data.
    alpha2   = zeros(alphasize,nt_); # azimuth data.
    alpha3   = zeros(alphasize,nt_); # dip data.
    Malpha1  = zeros(alphasize,nbins,nt_);
    Malpha2  = zeros(alphasize,nbins,nt_);
    Malpha3  = zeros(alphasize,nbins,nt_);

    rtd    = 180./pi;           # radians to degrees
    over90 = 1.0/90.;           # auxiliars

    for h = 1:nt_               # loop over traces.
        for k = begin1:end1-1   # loop over moving time window.
            # clear xp yp zp MP pP DP
            xP = tx[k-w:k+w,h];
            yP = ty[k-w:k+w,h];
            zP = tz[k-w:k+w,h];

            MP = covariance(xP,yP,zP);

            #---------------------------------------

            # DP: eigenvalues of covariance Matrix.
            DP =  eigvals(MP);

            # DPsp: Return a permutation vector of indices of DP that
            # puts it in ascending order: such that
            # DP[Dpsp[3]] > DP[Dpsp[2]] > DP[Dpsp[1]].
            DPsp = sortperm(DP);

            # lambda: sorted as lambda(1)<lambda(2)<lambda(3).
            # oposite to Jurkevics1988
            lambda = sort(DP, rev=false)

            # pP: Returns a matrix pP whose columns are the
            # eigenvectors of MP. (The kth eigenvector can be obtained
            # from the slice pP[:, k].) contains the eigenvectors.
            pP = eigvecs(MP); # eigenvectors of covarience Matrix

            #such that u_i: eigenvectors such that u3 > u2 > u1
            u3 = pP[:,DPsp[3]];
	    	u2 = pP[:,DPsp[2]];
            u1 = pP[:,DPsp[1]];
	    	if (sum(u3)>0)
				u3 = -1.0*u3;
	    	end
            #---------------------------------------
            # computing polarization attributes
            #---------------------------------------

            # alpha1 = rectilinearity of P, Jurkevics(1988) -> [0,1]:
            alpha1[k-begin1+1,h]  = 1.0-0.5*((lambda[1]+lambda[2])/(lambda[3]));

            #    0 < alpha1      < 1
            #    0 < alpha*nbins < nbins

            fl1a = nbins*alpha1[k-begin1+1,h];
            fl1  = unsafe_trunc(Int64,fl1a);
            if abs.(fl1-fl1a) > 1.0
                error("1 - this truncation in h= ",h," and k= ",k," is wrong.");
            end
            # floor of the bin
            fl1  = Int64(fl1);
            # roof of the bin
            ro1  = fl1 + 1;

            # alpha2 = azimuth of P, Jurkevics(1988) ->  (-90, 90]:
            # The strike of the direction of maximum polarization
			alpha2arg             =	(sign(u3[1])*u3[2])/(sign(u3[1])*u3[3]);
            alpha2[k-begin1+1,h]  = atan(alpha2arg)*rtd;

            #       -90 <  alpha2       < 90
            #       -1  < (alpha2/90)   < 1
            #        0  < (alpha2/90)+1 < 2
            #        0  < ((alpha2/90)+1)*(nbins/2) < nbins

            fl2a = ((alpha2[k-begin1+1,h]*over90)+1)*(nbins*0.5);
            fl2  = unsafe_trunc(Int64,fl2a);
            if abs.(fl2-fl2a) > 1.0
                error("2 - this truncation in h= ",h," and k= ",k," is wrong.")
            end
            # floor of the bin
            fl2 = Int64(fl2);
            # roof of the bin
            ro2  = fl2 + 1;

            # alpha3 = dipP,         Vidale ->  (-90, 90]:
            # The dip of the direction of maximum polarization.
			alpha3arg            = u3[1]/sqrt(u3[2]^2+u3[3]^2);
            alpha3[k-begin1+1,h] = atan(alpha3arg)*rtd;

            #         -90 <  alpha3                   < 90
            #         -1  < (alpha3/90)               < 1
			#          0  < (alpha3/90)+1             < 2
            #          0  < ((alpha3/90)+1)*(nbins/2) < nbins
            fl3a = ((alpha3[k-begin1+1,h]*over90)+1)*(nbins*0.5);
            fl3  = unsafe_trunc(Int64,fl3a);
            if abs.(fl3-fl3a) > 1.0
                error("3 - this truncation in h= ",h," and k= ",k," is wrong.");
            end
            # floor of the bin
            fl3 = Int64(fl3);
            # roof of the bin
            ro3 = fl3 + 1;

            # is the trace of MP at all times 'k'
            Tr = trace(MP);

            # At any time k,h and fl1, it contains the values of Tr
            # each column of fl1 (related to parameter=alpha) correspond
            # to a particular bin for my histogram, so this columns
            # contains anly the values of Tr who's alpha's are equal.

            Malpha1[k-begin1+1,ro1,h] = Tr;
            Malpha2[k-begin1+1,ro2,h] = Tr;
            Malpha3[k-begin1+1,ro3,h] = Tr;


        end
    end

    # we reduce the matrix to a 2D array by summing along
    # the time (attribute) dimension.
    Malpha1sum = zeros(nbins,nt_);
    Malpha2sum = zeros(nbins,nt_);
    Malpha3sum = zeros(nbins,nt_);

    Malpha1sum = sum(Malpha1,1); Malpha1sum = reshape(Malpha1sum,nbins,nt_);
    Malpha2sum = sum(Malpha2,1); Malpha2sum = reshape(Malpha2sum,nbins,nt_);
    Malpha3sum = sum(Malpha3,1); Malpha3sum = reshape(Malpha3sum,nbins,nt_);

#    figure;
#    pcolormesh(Malpha1sum)

    Ha1 = Malpha1sum';
    Ha2 = Malpha2sum';
    Ha3 = Malpha3sum';
    ### Histogram Normalization
    # A-Preallocations
    Ha1n = zeros(nt_,nbins);
    Ha2n = zeros(nt_,nbins);
    Ha3n = zeros(nt_,nbins);

    # B- find maximum of (all bins value) matrix rows
    Ha1max = m./sum(Ha1,2);
    Ha2max = m./sum(Ha2,2);
    Ha3max = m./sum(Ha3,2);
    # C-apply normalization
    for i = 1:nt_
        Ha1n[i,:] = Ha1max[i]*Ha1[i,:]
        Ha2n[i,:] = Ha2max[i]*Ha2[i,:]
        Ha3n[i,:] = Ha3max[i]*Ha3[i,:]
    end

    # ### Histogram Normalization
    # # A-Preallocations
    # Ha1n = zeros(nt_,nbins);
    # Ha2n = zeros(nt_,nbins);
    # Ha3n = zeros(nt_,nbins);
    # # B- find sum of (all bins value) matrix rows
    # sum1 = sum(Ha1,2);
    # sum2 = sum(Ha2,2);
    # sum3 = sum(Ha3,2);
    # # C- divide each row by the sum value of the row.
    # Ha1n = 0.5*(Ha1n./sum1);
    # Ha2n = 0.5*(Ha2n./sum2);
    # Ha3n = 0.5*(Ha3n./sum3);

    return Ha1n,Ha2n,Ha3n;
end
