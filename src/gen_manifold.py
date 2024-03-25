import numpy as np

def manifoldGen(manifoldType):
    D = 100  # ambient space dimension
    sigma = 0.001  # noise variance
    
    if manifoldType == '2trefoils':
        N = 100
        gtruth = np.concatenate((np.zeros(N), np.ones(N)))

        # Generate two trefoil-knots
        d = 3
        Par = 3.8
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        Yg = {}
        Yg[1] = np.array([(2+np.cos(3*t))*np.cos(2*t), (2+np.cos(3*t))*np.sin(2*t), np.sin(3*t)])
        Yg[2] = np.array([(2+np.cos(3*t))*np.cos(2*t) + Par, (2+np.cos(3*t))*np.sin(2*t), np.sin(3*t)])
        Y = np.concatenate((Yg[1], Yg[2]), axis=1)
        U = np.linalg.qr(np.random.randn(D, d))[0]
        Yn = np.dot(U, Y)
        Yn = Yn + sigma * np.random.randn(*Yn.shape)
        x = np.array([np.cos(t), np.cos(t), np.sin(t), np.sin(t)])
        
    elif manifoldType == 'sphere':
        N = 1000
        gtruth = np.ones(N)
        
        # Generate a random sphere
        d = 3
        r = 2 * (np.sort(np.random.rand(N))**0.99)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        p = np.random.permutation(N)
        P = np.zeros((N, N))
        for i in range(N):
            P[p[i], i] = 1
        theta = np.dot(theta, P)
        xx = r * np.cos(theta)
        yy = r * np.sin(theta)
        
        Y = np.array([2*xx/(1+xx**2+yy**2), 2*yy/(1+xx**2+yy**2), (-1+xx**2+yy**2)/(1+xx**2+yy**2)])
        U = np.linalg.qr(np.random.randn(D, d))[0]
        Yn = np.dot(U, Y)
        Yn = Yn + sigma * np.random.randn(*Yn.shape)
        x = np.array([xx, yy])
    
    else:
        raise ValueError('Unknown Manifold Type!')
    
    return Yn, Y, gtruth, x