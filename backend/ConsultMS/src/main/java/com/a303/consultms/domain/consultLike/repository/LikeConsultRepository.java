package com.a303.consultms.domain.consultLike.repository;

import com.a303.consultms.domain.consultLike.LikeConsult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface LikeConsultRepository extends JpaRepository<LikeConsult, Integer> {

    LikeConsult findByConsultIdAndMemberId(int consultId, int memberId);

    LikeConsult findByLikeConsultIdAndMemberId(int likeConsultId, int memberId);
}
